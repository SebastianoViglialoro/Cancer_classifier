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
        #definisco le colonne da normalizzare
        if exclude_col is None:
            exclude_col = []

        data_normalized = data.copy()
        for column in data.columns:
            if column not in exclude_col and pd.api.types.is_numeric_dtype(data[column]):
                mean_val = data[column].mean()
                std_val = data[column].std()
                data_normalized[column] = (data[column] - mean_val) / std_val
        return data_normalized

class Preprocessing:
    """
    Classe per gestire il preprocessing dei dati:
    - Normalizzazione (Min-Max o Standardizzazione)
    - Salvataggio del dataset normalizzato
    - Separazione delle feature dal target
    """
    @staticmethod
    def choose_normalization_method():
        """
        Chiede all'utente di scegliere il metodo di normalizzazione.
        """
        print("Scegli come normalizzare i dati:")
        print("Modalità disponibili: ['normalizzazione min-max', 'standardizzazione']")
        method = input("Inserisci la modalità di normalizzazione: ").strip().lower()

        if method not in ['normalizzazione min-max', 'standardizzazione']:
            print("Modalità non supportata. Verrà utilizzata la modalità di default: normalizzazione min-max.")
            method = 'normalizzazione min-max'
        
        return method

    @staticmethod
    def get_normalizer(data: pd.DataFrame, exclude_col: list) -> pd.DataFrame:
        """
        Normalizza i dati utilizzando Min-Max o Standardizzazione (Z-score).
        """
        method = Preprocessing.choose_normalization_method()

        if exclude_col is None:
            exclude_col = []

        if method == 'normalizzazione min-max':
            return MinMaxNormalizer.normalize(data, exclude_col)
        if method == 'standardizzazione':
            return StandardNormalizer.standardize(data, exclude_col)
        else:
            raise ValueError("Normalizzazione non supportata. Usare 'minmax' o 'standard'")
        
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

    @staticmethod
    def split_features_target(data: pd.DataFrame, target_col: str):
        """
        Separa le features dalla colonna target.
        """
        if target_col not in data.columns:
            raise ValueError(f"La colonna target '{target_col}' non è presente nel dataset.")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        return X, y