from data_cleaning import SelectionFile, DataCleaner
from data_cleaning import Preprocessing
from models import Modelling
from evaluation import Validation
from utils import get_valid_int
from utils import select_metrics

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
    try:
        data_scaled = Preprocessing.get_normalizer(cleaned_data, exclude_col=['Sample code number', 'classtype_v1'])
    except Exception as e:
        print(f"Errore durante la normalizzazione: {e}. Uso i dati puliti senza normalizzazione.")
        data_scaled = cleaned_data

    #Salvataggio del dataset normalizzato
    data_scaled = Preprocessing.save_dataset(data_scaled)

    #Separazione feature (X) e target (y)
    X, y = Preprocessing.split_features_target(data_scaled, target_col='classtype_v1')

    #Scelta del numero di vicini k
    k = get_valid_int("Inserisci il numero di vicini (k) per il classificatore k-NN: ", min_value=1)

    #Creazione del modello k-NN tramite Modelling
    model_m = Modelling(model_type="knn", k=k)

    #Scelta del numero di folds per K-Fold Cross Validation
    num_folds = get_valid_int("Inserisci il numero di folds per la K-Fold cross validation: ", min_value=2)

    # Selezione delle metriche di valutazione
    selected_metrics = select_metrics()

    #Creazione dell'istanza di Validation
    validator = Validation(classifier=model_m, X=X, y=y, num_folds=num_folds, selected_metrics=selected_metrics)

    #Esecuzione della validazione K-Fold e salvataggio risultati
    validator.k_fold_cross_validation()
    

if __name__ == "__main__":
    main()
