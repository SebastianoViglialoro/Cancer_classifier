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

class SaveNormDB:
    @staticmethod
    def save_dataset(data: pd.DataFrame):
        save_option = input("\nVuoi salvare il dataset normalizzato? (s/n): ").strip().lower()
        if save_option == 's':
            default_folder = "data/scaled"  # Cartella di default
            os.makedirs(default_folder, exist_ok=True)  # Crea la cartella se non esiste

            output_filename = input(f"Inserisci il nome del file di output (lascia vuoto per 'scaled_data.csv'): ").strip()

            # Se non viene specificato un nome, usa "scaled_data.csv"
            if not output_filename:
                output_filename = "scaled_data.csv"

            # Costruisci il percorso completo (relativo)
            relative_path = os.path.join(default_folder, output_filename)

            # Converti in percorso assoluto
            absolute_path = os.path.abspath(relative_path)

            # Salva il file
            data.to_csv(absolute_path, index=False)
            print(f"Dataset pulito salvato in: {absolute_path}")

class StandardNormalizer:
    """
    Questa classe permette di normalizzare i dati in un DataFrame utilizzando la standardizzazione (z-score)
    """
    @staticmethod
    def standardize(data: pd.DataFrame, exclude_col: list) -> pd.DataFrame:
        """
        Normalizza i dati, nel range [0,1], in un DataFrame utilizzando la formula:
        x = (x - mean) / std
        """
        #definisco le colonne da normalizzare se non ci sono colonne escluse
        if exclude_col is None:
            exclude_col = []

        data_normalized = data.copy()
        for column in data.columns:
            if column not in exclude_col and pd.api.types.is_numeric_dtype(data[column]):
                mean_val = data[column].mean()
                std_val = data[column].std()
                data_normalized[column] = (data[column] - mean_val) / std_val
        return data_normalized
    
class SelectNormalizer:
    """
    Questa classe seleziona il tipo di normalizzazione da applicare ai dati
    """
    @staticmethod
    def get_normalizer(normalizer: str, data: pd.DataFrame, exclude_col: list) -> MinMaxNormalizer:
        if normalizer == 'normalizzazione min-max':
            return MinMaxNormalizer.normalize(data, exclude_col)
        if normalizer == 'standardizzazione':
            return StandardNormalizer.standardize(data, exclude_col)
        else:
            raise ValueError("Normalizzazione non supportata. Usare 'minmax' o 'standard'")