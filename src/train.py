import warnings
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, learning_curve
from xgboost import XGBClassifier

import src.constants as cte
from src.data import split_data
from src.evaluate import best_f1_threshold, evaluate
from src.preprocessing import preprocessing, build_preprocessor
from src.utils import save_models


REPORTS_DIR = Path("reports")
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def fit_and_evaluate(name, model, data, cv):
    """CV scores, out-of-fold threshold, final fit, and train/val/test evaluation."""
    X_train, y_train = data["train"]

    # Train vs CV scores (overfitting check)
    s = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=True)
    cv_results = {
        "model": name,
        "train_pr_auc": s["train_pr_auc"].mean(),
        "cv_pr_auc": s["test_pr_auc"].mean(),
        "gap_pr_auc": s["train_pr_auc"].mean() - s["test_pr_auc"].mean(),
        "train_roc_auc": s["train_roc_auc"].mean(),
        "cv_roc_auc": s["test_roc_auc"].mean(),
    }
    if cv_results["gap_pr_auc"] > cte.MAX_GAP:
        warnings.warn(f"{name}: PR-AUC gap {cv_results['gap_pr_auc']:.3f} > {cte.MAX_GAP}. Possible overfitting.")

    # Threshold from out-of-fold predictions
    oof_prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof_prob)

    # Final fit and train/val/test evaluation
    model.fit(X_train, y_train)
    results = [
        evaluate(name, split, y, model.predict_proba(X)[:, 1], threshold)
        for split, (X, y) in data.items()
    ]
    return threshold, cv_results, results

def main():
    # 1. Clean data and engineer features
    df = preprocessing()

    # 2. Split data
    X_train_pre, X_val_pre, X_test_pre, y_train, y_val, y_test = split_data(df)

    # 3. Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train_pre)

    # 4. Fit ONLY on training data and transform validation and test
    data = {
        "train": (preprocessor.fit_transform(X_train_pre), y_train),
        "val": (preprocessor.transform(X_val_pre), y_val),
        "test": (preprocessor.transform(X_test_pre), y_test),
    }

    # 5. Train the model
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=cte.N_SPLITS, shuffle=True, random_state=cte.RANDOM_STATE)

    # Hyperparameters justified in notebook/02_model_training_and_evaluation.ipynb
    models = {
        f"Logistic Regression(C={cte.C})": LogisticRegression(C=cte.C, max_iter=2000, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            scale_pos_weight=scale_pos_weight, subsample=0.7, colsample_bytree=0.7,
            random_state=cte.RANDOM_STATE, **cte.XGB_CONFIG,
        ),
    }
    
    thresholds = {}
    cv_results = []
    results = []
    for name, model in models.items():
        thresholds[name], cv_res, res = fit_and_evaluate(name, model, data, cv)
        cv_results.append(cv_res)
        results.extend(res)
        
    # 6. Report
    REPORTS_DIR.mkdir(exist_ok=True)
    cv_df, results_df = pd.DataFrame(cv_results), pd.DataFrame(results)
    cv_df.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    results_df.to_csv(REPORTS_DIR / "final_results.csv", index=False)
    print(cv_df.round(4), "\n")
    print(results_df.round(4))
    
    # 7. Save models
    logreg, xgb = models.values()
    thr_lr, thr_xgb = thresholds.values()
    save_models(logreg, xgb, preprocessor, thr_lr, thr_xgb)
    
    return {
        "models": models,
        "preprocessor": preprocessor,
        "thresholds": thresholds,
        "cv_results": cv_df, 
        "results": results_df
    }

    
if __name__ == "__main__":
    main()