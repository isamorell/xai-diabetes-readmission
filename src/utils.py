from pathlib import Path
import joblib

MODELS_DIR = Path("models")

def save_models(logreg, xgb, preprocessor, threshold_lr, threshold_xgb):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = {
        "logistic_regression": threshold_lr,
        "xgboost": threshold_xgb,
    }

    joblib.dump(logreg, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(thresholds, MODELS_DIR / "thresholds.joblib")
    
def load_models():
    logreg = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    thresholds = joblib.load(MODELS_DIR / "thresholds.joblib")
    
    return logreg, xgb, preprocessor, thresholds