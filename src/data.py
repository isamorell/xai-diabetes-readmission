import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 

import src.constants as cte


def load_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    # fetch dataset 
    diabetes_us_hospitals: dict = fetch_ucirepo(id=296) 

    X: pd.DataFrame = diabetes_us_hospitals.data.features 
    y: pd.DataFrame = diabetes_us_hospitals.data.targets 
    return diabetes_us_hospitals, X, y

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    """Separate features, target and identifiers, then split the dataset."""
    X = df.drop(columns=[cte.TARGET] + cte.ID_COLS)
    y = df[cte.TARGET]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=cte.SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=cte.SEED
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
