from abc import ABC, abstractmethod
import pandas as pd

class Fileimporter(ABC):
    """
    Classe astratta per l'importazione di dati da diversi formati di file    
    """
    @abstractmethod
    def importer(self, file_path: str) -> pd.DataFrame:
        pass

#CSV
class FileCSV(Fileimporter):
    """
    Questa classe implementa l'import per File CSV
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_csv(file_path)
        #rimuovere duplicati?
        return df 

#Excel
class FileExcel(Fileimporter):
    """
    Questa classe implementa l'import per File Excel
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_excel(file_path)
        #rimuovere duplicati?
        return df 

#TSV
class FileTSV(Fileimporter):
    """
    Questa classe implementa l'import per File TSV
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_csv(file_path)
        #rimuovere duplicati?
        return df 

#TXT
class FileTXT(Fileimporter):
    """
    Questa classe implementa l'import per File TXT
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_csv(file_path)
        #rimuovere duplicati?
        return df 

#JSON
class FileJSON(Fileimporter):
    """
    Questa classe implementa l'import per File JSON
    """
    def importer(self, file_path: str) -> pd.DataFrame:
        print(f"Import del file: {file_path}")
        df = pd.read_json(file_path)
        #rimuovere duplicati?
        return df 

#Permette di selezionare il metodo di import in base al tipo di file dell'utente
class SelectionFile:
    """
    Questa classe seleziona il tipo di file da importare
    """
    @staticmethod

    def get_file(file_path: str) -> Fileimporter:
        if file_path.endswith('.csv'):
            return FileCSV()
        elif file_path.endswith('.xlsx'):
            return FileExcel()
        elif file_path.endswith('.tsv'):
            return FileTSV()
        elif file_path.endswith('.txt'):
            return FileTXT()
        elif file_path.endswith('.json'):
            return FileJSON()
        else:
            raise ValueError("Formato file non supportato, usare un file con estenzione [.csv, .xlsx, .tsv, .txt, .json]")