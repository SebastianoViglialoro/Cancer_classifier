import os 
import pandas as pd
from data_cleaning import SelectionFile, GestioneValMancanti, SaveDB
from data_cleaning import SelectNormalizer, SaveNormDB
from models import m_knn

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

    # 2: All'utente viene posto di scegliere come gestire i valori mancanti
    print("Scegliere come gestire i valori mancanti attraverso le modalità sviluppate.")
    print("Modalità disponibili: ['rimozione', 'media', 'moda', 'mediana']")
    mode = input("Inserisci la modalità di gestione dei valori mancanti che vuoi usare: ").strip().lower()

    if mode not in ['rimozione', 'media', 'moda', 'mediana']: #se la modalità non è supportata, viene utilizzata la modalità di default: media
        print("Modalità non supportata. Verrà utilizzata la modalità di default: media")
        mode = 'media'
    
    try:
        data = GestioneValMancanti.get_mode(mode, data) #richiamo i metodi per la gestione dei valori mancanti
    except Exception as e:
        print(f"Errore durante la gestione dei valori mancanti: {e}. Procedo con i dati originali.")

    print("Dati dopo la gestione dei valori mancanti:")
    print(data.head()) #mostra le prime 5 righe del dataset dopo la gestione dei valori mancanti
    print(f"\n Controllo se il dataset contiene ancora valori nulli nella colonna: {data.isnull().sum()}")


    # Separazione feature (X) e target (y)
    X = data.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
    y = data.iloc[:, -1].values   # Ultima colonna come etichette

    # Scelta dei parametri del modello knn
    while True:
        try:
            k = int(input("Inserisci il numero di vicini (k) per il classificatore k-NN: "))
            if k <= 0:
                raise ValueError("k deve essere un numero intero positivo.")
            break
        except ValueError as e:
            print(f"Errore: {e}. Riprova.")
    classifier = m_knn.KNNClassifier(k=k)


if __name__ == "__main__":
    main()