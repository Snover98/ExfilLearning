import pandas as pd
from sklearn import preprocessing
from typing import List


class DFScaler:
    def __init__(self, df: pd.DataFrame, columns: List[str], scaler_class=preprocessing.StandardScaler):
        self.columns = columns
        self.scaler = scaler_class()
        self.scaler.fit(df[self.columns])

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df
