import os 
import pandas as pd
from data_cleaning import SelectionFile, GestioneValMancanti, SaveDB
from data_cleaning import SelectNormalizer, SaveNormDB

def main():
    # 1: Input dell'utente per l'import del file
    ok = False
    while not ok:
        file_path = input("Inserisci il percorso del file del dataset di analisi: ").strip()
        if not file_path:
            print("Il percorso non può essere vuoto. Riprova.")
            continue  # Torna all'inizio del ciclo
        print("Import del dataset in corso...")

        try:
            importer = SelectionFile.get_file(file_path)
            data = importer.importer(file_path)
            break  # Esce dal loop solo se il file viene caricato correttamente
        except Exception as e:
            print(f"Errore durante l'import del file: \n{e}. Riprova.")

    if data.empty: #se il dataset è vuoto, termina l'esecuzione e arresta il sistema
        print("Il dataset è vuoto. Termino l'esecuzione.")
        return

    print("Dati originali:")
    print(data.head()) #mostra le prime 5 righe del dataset
