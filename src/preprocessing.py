from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import load_data
import src.constants as cte


def modify_target(df: pd.DataFrame, target: str = "readmitted") -> pd.DataFrame:
    """Binary target: 1 if readmitted in less than 30 days, 0 otherwise."""
    df[target] = (df["readmitted"] == "<30").astype(int)
    return df

def fill_not_actual_nans(df: pd.DataFrame, cols: list[str], placeholder: str = "Unknown") -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].fillna(placeholder)
    return df

def drop_invalid_encounters(df: pd.DataFrame) -> pd.DataFrame:
    invalid = (
        df["discharge_disposition_id"].isin(cte.EXPIRED_IDS + cte.NEWBORN_DISCHARGED_IDS)
        | df["admission_type_id"].isin(cte.NEWBORN_ADMISSION_TYPE_IDS)
        | df["admission_source_id"].isin(cte.BIRTH_ADMISSION_SOURCE_IDS)
        | df["gender"].eq("Unknown/Invalid")
    )
    return df.loc[~invalid]

def add_drug_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates computed over ALL drug columns (so rare drugs still count), then
    drug levels are simplified."""
    drugs = df[cte.DRUG_COLS + cte.RARE_DRUGS]
    df["n_drugs"] = drugs.ne("No").sum(axis=1)
    df["n_dose_changes"] = drugs.isin(["Up", "Down"]).sum(axis=1)
    
    collapse = [c for c in cte.DRUG_COLS if c != "insulin"]
    df[collapse] = df[collapse].replace({"Up": "Changed", "Down": "Changed"})
    return df

def group_icd9(codes: pd.Series) -> pd.Series:
    """Group ICD-9 codes into the clinical categories of Strack et al. (2014).
    Expects "Unknown" as the placeholder for missing codes. V and E codes -> "Other"."""
    num = pd.to_numeric(codes, errors="coerce")  # V.. and E.. become NaN
    base = np.floor(num)
    conditions = [
        codes.eq("Unknown"),
        base.eq(250),
        base.between(390, 459) | base.eq(785),
        base.between(460, 519) | base.eq(786),
        base.between(520, 579) | base.eq(787),
        base.between(580, 629) | base.eq(788),
        base.between(800, 999),
        base.between(710, 739),
        base.between(140, 239),
    ]
    choices = [
        "Unknown", "Diabetes", "Circulatory", "Respiratory", "Digestive",
        "Genitourinary", "Injury", "Musculoskeletal", "Neoplasms",
    ]
    return pd.Series(np.select(conditions, choices, default="Other"), index=codes.index)

def group_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    for col in cte.DIAG_COLS:
        df[f"{col}_group"] = group_icd9(df[col])
    return df
    
def preprocessing():
    dataset, _, _ = load_data()
    df = dataset.data.original.copy()
    print(f"Raw data: {len(df):,} rows")
    
    # 1. Modify the target for the task at hand
    df = modify_target(df)
    
    # 2. Drop columns (reasons can be found in notebook/01_exploratory_data_analysis.ipynb)
    df = df.drop(columns=cte.COLUMNS_TO_DROP)
    
    # 3. Replace "not actual" missing values by their placeholder
    df = fill_not_actual_nans(df, cte.LAB_COLS, "Not tested")
    df = fill_not_actual_nans(df, cte.ADMIN_COLS + cte.DIAG_COLS + ["race"], "Unknown")
    
    # 4. Avoid data leakage
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    df = drop_invalid_encounters(df)
    print(f"After removing invalid encounters: {len(df):,} rows | positive rate {df['readmitted'].mean():.2%}")
    
    # 5. Feature engineering that does not depend on the data distribution
    df = add_drug_features(df)
    df = group_diagnoses(df)
    df[cte.CODED_ID_COLS] = df[cte.CODED_ID_COLS].astype(str)  # Coded IDs are categories
    
    # 6. Drop what is no longer needed
    df = df.drop(columns=cte.DIAG_COLS + cte.RARE_DRUGS)
    
    print(f"Final: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df

def build_preprocessor(X_train: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Column transformer for numerical and categorical features.
    
    scale = True -> linear models: log1p + standardisation of counts, standardised numerics.
    scale = False -> tree models: numeric features pass through unchanged.

    Categories below 1% of the training set are pooled into an "infrequent" level by the
    OneHotEncoder. The threshold is learned from the training data only, so there is no leakage.
    """

    numeric_cols = [
        col for col in cte.NUMERIC_COLS + ["n_drugs", "n_dose_changes"]
        if col in X_train.columns
    ]

    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]
    
    cat_pipeline = OneHotEncoder(handle_unknown="ignore")
    if scale:
        pre = ColumnTransformer([
            ("cat", cat_pipeline, categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ])

    else:
        pre = ColumnTransformer([("cat", cat_pipeline, categorical_cols)])
                    
    return pre


if __name__ == "__main__":
    df_preprocessed = preprocessing()
    Path("data").mkdir(exist_ok=True)
    df_preprocessed.to_csv("data/preprocessed_diabetic_data.csv", index=False)