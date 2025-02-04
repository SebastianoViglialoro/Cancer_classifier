import os 
import pandas as pd
from data_cleaning import SelectionFile, GestioneValMancanti, DataCleaner
from data_cleaning import SelectNormalizer, SaveNormDB
from models.model_management import Modelling
from evaluation.model_evalutation import Validation
from utils import input_valid_int

def main():
    #Input dell'utente per l'import del file
    data = SelectionFile.import_data()

    if data.empty: #se il dataset è vuoto, termina l'esecuzione e arresta il sistema
        print("Il dataset è vuoto. Termino l'esecuzione.")
        return

    print("Dati originali:")
    print(data.head()) #mostra le prime 5 righe del dataset

    #All'utente viene posto di scegliere come gestire i valori mancanti
    cleaned_data = DataCleaner.clean_and_save(data)  # Pulisce i dati e salva il dataset

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
        data_scaled= SelectNormalizer.get_normalizer(norm_mode, cleaned_data, exclude_col)
    except Exception as e:
        print(f"Errore durante la gestione dei valori mancanti: {e}. Procedo con i dati originali.")
        data_scaled = data
    
    print("Dati dopo la normalizzazione:")
    print(data_scaled.head())

    #Salvataggio del dataset normalizzato nella cartella data/normalized
    SaveNormDB.save_dataset(data_scaled)

    #Separazione feature (X) e target (y)
    X = data_scaled.drop(columns=['classtype_v1']).values  # Feature
    y = data_scaled['classtype_v1'].values   # Etichette

    #Scelta del numero di vicini k
    k = input_valid_int("Inserisci il numero di vicini (k) per il classificatore k-NN: ", min_value=1)

    #Creazione del modello k-NN tramite Modelling
    model_m = Modelling(model_type="knn", k=k)

    #Scelta del numero di folds per K-Fold Cross Validation
    num_folds = input_valid_int("Inserisci il numero di folds per la K-Fold cross validation: ", min_value=2)

    #Creazione dell'istanza di Validation
    validator = Validation(model=model_m, X=X, y=y, num_folds=num_folds)

    #Esecuzione della validazione K-Fold e salvataggio risultati
    validator.k_fold_cross_validation()
    

if __name__ == "__main__":
    main()
