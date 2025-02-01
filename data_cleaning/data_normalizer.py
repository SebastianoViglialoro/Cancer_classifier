import pandas as pd
import os

class MinMaxNormalizer:
    """
    Questa classe permette di normalizzare i dati in un DataFrame utilizzando la normalizzazione Min-Max 
    """
    @staticmethod
    def normalize(data: pd.DataFrame, exclude_col: list) -> pd.DataFrame:
        """
        Normalizza i dati, nel range [0,1], in un DataFrame utilizzando la formula:
        x = (x - min) / (max - min)
        """
        #definisco le colonne da normalizzare
        if exclude_col is None:
            exclude_col = []

        data_normalized = data.copy()
        for column in data.columns:
            if column not in exclude_col and pd.api.types.is_numeric_dtype(data[column]):
                min_val = data[column].min()
                max_val = data[column].max()
                data_normalized[column] = (data[column] - min_val) / (max_val - min_val)
        return data_normalized
    
