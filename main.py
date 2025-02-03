import os 
import pandas as pd
from data_cleaning import SelectionFile, GestioneValMancanti, SaveDB
from data_cleaning import SelectNormalizer, SaveNormDB
from models import KNNClassifier
from evaluation import ModelEvaluation

def main():
    #Input dell'utente per l'import del file
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

    #All'utente viene posto di scegliere come gestire i valori mancanti
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

    #Salvataggio del dataset pulito nella cartella data/cleaned
    SaveDB.save_dataset(data)

    #Normalizzazione dei dati
    print("Scegliere come normalizzare i dati attraverso le funzioni sviluppate.")
    print("Modalità disponibili: ['normalizzazione min-max', 'standardizzazione']")
    norm_mode = input("Inserisci la modalità di gestione dei valori mancanti che vuoi usare: ").strip().lower()

    if norm_mode not in ['normalizzazione min-max', 'standardizzazione']: #se la modalità non è supportata, termina l'esecuzione
        print("Modalità non supportata. Verrà utilizzata la modalità di default: normalizzazione min-max")
        norm_mode = 'normalizzazione min-max'
    
    data_scaled = None
    exclude_col = ['Sample code number','classtype_v1'] #escludiamo le colonna target e la colonna dei campioni(rappresentano l'id del campione)
    try:
        data_scaled= SelectNormalizer.get_normalizer(norm_mode, data, exclude_col)
    except Exception as e:
        print(f"Errore durante la gestione dei valori mancanti: {e}. Procedo con i dati originali.")
        data_scaled = data
    
    print("Dati dopo la normalizzazione:")
    print(data_scaled.head())

    #Salvataggio del dataset normalizzato nella cartella data/normalized
    SaveNormDB.save_dataset(data_scaled)

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
    classifier = KNNClassifier(k=k)

    # Esecuzione della validazione K-Fold
    num_folds = int(input("Inserisci il numero di folds per la K-Fold cross validation: "))
    evaluator = ModelEvaluation(classifier, X, y, num_folds=num_folds)
    results_df = evaluator.k_fold_cross_validation()
    results_df.to_csv("results/k-fold/k_fold_results.csv", index=False)
    print("\n Risultati della validazione K-Fold salvati in results/k-fold/k_fold_results.csv")


if __name__ == "__main__":
    main()
