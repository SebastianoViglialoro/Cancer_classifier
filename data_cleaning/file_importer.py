from abc import ABC, abstractmethod
import pandas as pd

class Fileimporter(ABC):
    """
    Abstract class for importing data from different file formats
    """
    @abstractmethod
    def importer(self, file_path: str) -> pd.DataFrame:
        pass
    
#CSV ---
class FileCSV(Fileimporter):
    """
    Questa classe implementa l'import per File CSV
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_csv(file_path)
        return df 

#EXCEL --- 
class FileExcel(Fileimporter):
    """
    Questa classe implementa l'import per File Excel
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_excel(file_path)
        return df 

#TSV --- 
class FileTSV(Fileimporter):
    """
    Questa classe implementa l'import per File TSV
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_csv(file_path)
        return df 
    
#TXT --- 
class FileTXT(Fileimporter):
    """
    Questa classe implementa l'import per File TXT
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_csv(file_path)
        return df 

#JSON ---
class FileJSON(Fileimporter):
    """
    Questa classe implementa l'import per File JSON
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_json(file_path)
        return df 


class SelectionFile:
    """
    Gestisce la selezione e l'importazione dei file in base all'estensione.
    """
    @staticmethod
    def get_importer(file_path: str) -> Fileimporter:
        if file_path.endswith('.csv'):
            return FileCSV()
        elif file_path.endswith(('.xlsx', '.xls')):
            return FileExcel()
        elif file_path.endswith('.tsv'):
            return FileTSV()
        elif file_path.endswith('.txt'):
            return FileTXT()
        elif file_path.endswith('.json'):
            return FileJSON()
        else:
            raise ValueError("Formato file non supportato. Usa uno di questi formati: [.csv, .xlsx, .tsv, .txt, .json]")

    @staticmethod
    def import_data() -> pd.DataFrame:
        """
        Richiede all'utente di inserire il percorso del file e gestisce eventuali errori di importazione.
        """
        ok = False
        while not ok:
            file_path = input("Inserisci il percorso del file del dataset di analisi: ").strip()

            if not file_path:
                print("Il percorso non può essere vuoto. Riprova.")
                continue

            try:
                importer = SelectionFile.get_importer(file_path)
                data = importer.importer(file_path)
                print("Importazione completata con successo!")
                return data  # Uscita dal ciclo se l'importazione ha successo
            except Exception as e:
                print(f"Errore durante l'import del file: {e}")
                print("Riprova inserendo un percorso valido.")