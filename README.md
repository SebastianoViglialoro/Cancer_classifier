---------------------------------------------------------
Cancer Classifier - k-NN
---------------------------------------------------------

-Descrizione
  Questo progetto implementa un classificatore di tumori basato su k-Nearest Neighbors (k-NN) con validazione K-Fold. Il modello viene addestrato su dati normalizzati e puliti, e al termine della validazione genera grafici di analisi, come la matrice di confusione e la curva ROC-AUC.

-Approccio adottato

  Abbiamo implementato il progetto seguendo un'architettura modulare e scalabile, suddividendo il codice in diverse componenti:

      *Preprocessing dei dati: Gestione dei valori mancanti e normalizzazione dei dati.

      *Modellizzazione: Implementazione del modello k-NN senza l'uso di Scikit-Learn, per comprendere a fondo l'algoritmo.

      *Validazione: Utilizzo di K-Fold Cross Validation per garantire una valutazione robusta delle prestazioni del modello.

      *Visualizzazione: Generazione automatica della matrice di confusione e della curva ROC-AUC per l’analisi della          performance del modello.

      *Abbiamo evitato l’uso di librerie preconfezionate per il machine learning, implementando manualmente i metodi di classificazione e validazione.

-Installazione

1️- Clona il repository

git clone https://github.com/tuo-username/Cancer_classifier.git
cd Cancer_classifier

2️- Installa le dipendenze

pip install -r requirements.txt

3- Assicurati che il dataset sia presente

Il dataset deve essere posizionato nella cartella data/.
Se non è presente, scaricalo e posizionalo come segue:

📂 Cancer_classifier/
 ├── 📂 data/
 │    ├── version_1.csv  → (Dataset da utilizzare)  

-Esecuzione
Per avviare il classificatore, eseguire:

python main.py

Il progr amma chiederà:
    *Il percorso del dataset
    *Come gestire i valori mancanti
    *Come normalizzare i dati
    *Il numero di vicini k
    *Il numero di fold per la K-Fold Cross Validation

Una volta completata l’esecuzione, il modello verrà valutato e saranno generati i grafici di analisi.

-Output e Risultati

I risultati della validazione vengono salvati nella cartella:

📂 results/k-fold/
 ├── k_fold_results.csv → Contiene i risultati della validazione K-Fold

-Analisi dei risultati

Al termine dell'esecuzione, il codice genera grafici di valutazione:
    *Matrice di Confusione → Mostra il numero di predizioni corrette ed errate.
    *Curva ROC-AUC → Valuta la capacità del modello nel distinguere le classi.


-Struttura del Progetto

📂 Cancer_classifier/
 ├── 📂 data_cleaning/         # Preprocessing e gestione dati
 │    ├── file_loader.py
 │    ├── data_cleaning.py
 │    ├── normalizzazione.py
 ├── 📂 models/                # Modello k-NN
 │    ├── m_knn.py
 ├── 📂 evaluation/             # Valutazione e grafici
 │    ├── __init__.py
 │    ├── model_evaluation.py
 │    ├── visualization.py
 ├── 📂 results/                # Risultati delle validazioni
 │    ├── k-fold/
 │    │    ├── k_fold_results.csv
 ├── README.md                 # Documentazione del progetto
 ├── requirements.txt          # Dipendenze del progetto
 ├── main.py                   # Script principale



