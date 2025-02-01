import pandas as pd 
import os 

class RimuoviDuplicati:
    """
    Questa Classe permette di rimuovere righe duplicate da un DataFrame
    """
    @staticmethod
    def dup_remove(data: pd.DataFrame) -> pd.DataFrame:
        data_cleaned = data.drop_duplicates()
        return data_cleaned

class ValoriMancanti:
    """
     Questa Classe permette di gestire i valori mancanti in un DataFrame a seconda di una modalità scelta dall'utente
    """
    @staticmethod
    def rimuovi_righe_classtype_v1(data: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove le righe che hanno un valore mancante nella colonna 'classtype_v1'. 
        """
        return data.dropna(subset=['classtype_v1'])
    """
    L'utente può scegliere tra diverse modalità per gestire i valori mancanti:
    """
    @staticmethod
    def rimuovi_righe_con_nan(data: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove tutte le righe che contengono valori NaN.
        """
        return data.dropna()

    @staticmethod
    def sostituisci_con_media(data: pd.DataFrame) -> pd.DataFrame:
        """
        Sostituisce i valori mancanti con la media della colonna.
        """
        return data.fillna(data.mean(numeric_only=True))

    @staticmethod
    def sostituisci_con_moda(data: pd.DataFrame) -> pd.DataFrame:
        """
        Sostituisce i valori mancanti con la moda della colonna.
        """
        return data.fillna(data.mode().iloc[0])

    @staticmethod
    def sostituisci_con_mediana(data: pd.DataFrame) -> pd.DataFrame:
        """
        Sostituisce i valori mancanti con la deviazione standard della colonna.
        """
        return data.fillna(data.median())
    
class SaveDB:
    @staticmethod
    def save_dataset(data: pd.DataFrame):
        save_option = input("\nVuoi salvare il dataset pulito? (s/n): ").strip().lower()
        if save_option == 's':
            default_folder = "data/cleaned"  # Cartella di default
            os.makedirs(default_folder, exist_ok=True)  # Crea la cartella se non esiste

            output_filename = input(f"Inserisci il nome del file di output (lascia vuoto per 'cleaned_data.csv'): ").strip()

            # Se non viene specificato un nome, usa "cleaned_data.csv"
            if not output_filename:
                output_filename = "cleaned_data.csv"

            # Costruisci il percorso completo (relativo)
            relative_path = os.path.join(default_folder, output_filename)

            # Converti in percorso assoluto
            absolute_path = os.path.abspath(relative_path)

            # Salva il file
            data.to_csv(absolute_path, index=False)
            print(f"Dataset pulito salvato in: {absolute_path}")
    
class GestioneValMancanti:
    """
    Questa Classe permette di gestire i valori mancanti in un DataFrame a seconda di una modalità scelta dall'utente
    """

    @staticmethod
    def get_mode(mode: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Funzione per gestire i valori nulli di un DataFrame
        """
        data = RimuoviDuplicati.dup_remove(data)

        data = ValoriMancanti.rimuovi_righe_classtype_v1(data)

        if mode == "rimozione":
            return ValoriMancanti.rimuovi_righe_con_nan(data)
        elif mode == "media":
            return ValoriMancanti.sostituisci_con_media(data)
        elif mode == "moda":
            return ValoriMancanti.sostituisci_con_moda(data)
        elif mode == "mediana":
            return ValoriMancanti.sostituisci_con_mediana(data)  
        else:
            raise ValueError("Modalità non supportata. Usare una modalità tra: ['rimozione', 'media', 'moda', 'mediana']")  



