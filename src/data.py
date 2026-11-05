import pandas as pd

from ucimlrepo import fetch_ucirepo 


def load_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    # fetch dataset 
    diabetes_us_hospitals: dict = fetch_ucirepo(id=296) 

    X: pd.DataFrame = diabetes_us_hospitals.data.features 
    y: pd.DataFrame = diabetes_us_hospitals.data.targets 
    return diabetes_us_hospitals, X, y
