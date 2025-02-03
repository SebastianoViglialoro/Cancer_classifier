---------------------------------------------------------
Cancer Classifier - k-NN
---------------------------------------------------------

-Descrizione

La classificazione delle cellule tumorali è essenziale per una diagnosi accurata e tempestiva. Esse si suddividono in due categorie principali: benigne, che rimangono circoscritte e non invadono altri tessuti, e maligne, caratterizzate da una crescita incontrollata con capacità di metastatizzare. L’analisi morfologica e biologica di queste cellule, attraverso parametri come la forma, le dimensioni e la struttura nucleare, consente di distinguerle e di supportare le decisioni cliniche.  

Questo progetto ha lo scopo di sviluppare un modello di intelligenza artificiale per classificare le cellule tumorali, basandosi sul k-Nearest Neighbors (k-NN). L'algoritmo opera su un dataset opportunamente trattato, garantendo la pulizia e la normalizzazione dei dati per migliorare l'affidabilità delle previsioni. Per assicurare una valutazione solida del modello, vengono adottate strategie di validazione incrociata, tra cui il metodo K-Fold, che consente di stimare con precisione le prestazioni del classificatore.  

Il sistema offre un alto grado di personalizzazione, permettendo agli utenti di scegliere diversi parametri, come la gestione dei dati mancanti, il metodo di normalizzazione e il numero di vicini da considerare. Alla fine del processo, vengono prodotti report dettagliati e rappresentazioni grafiche, tra cui la matrice di confusione e la curva ROC-AUC, strumenti fondamentali per valutare l’efficacia del modello e il suo potenziale utilizzo in ambito medico.


- Approccio adottato:

  Abbiamo implementato il progetto seguendo un'architettura modulare e scalabile, suddividendo il codice in diverse componenti:

  - **Preprocessing dei dati**: Gestione dei valori mancanti e normalizzazione dei dati.

  - **Modellizzazione**: Implementazione del modello k-NN senza l'uso di Scikit-Learn, per comprendere a fondo l'algoritmo.

  - **Validazione**: Utilizzo di K-Fold Cross Validation per garantire una valutazione robusta delle prestazioni del modello.

  - **Visualizzazione**: Generazione automatica della matrice di confusione e della curva ROC-AUC per l’analisi della performance del modello.

  - **Abbiamo evitato l’uso di librerie preconfezionate per il machine learning, implementando manualmente i metodi di classificazione e validazione**.
  

- Installazione
   
   1️. Clona il repository: 

    `git clone https://github.com/tuo-username/Cancer_classifier.git`

    `cd Cancer_classifier`

  2️. Installa le dipendenze

    `pip install -r requirements.txt`
  
  3. Assicurati che il dataset sia presente


- Installazione per Mac

    1. Genera un token di accesso personale (PAT):

    Vai su GitHub → Settings → Developer settings → Personal access tokens e crea un nuovo token con i permessi per repo.

    2. Configura Git con il token imposta il tuo nome utente ed email:

    `git config --global user.name "IlTuoNomeUtente"`
    `git config --global user.email "laTuaEmail@esempio.com"`

    `git remote set-url origin https://<TOKEN>@github.com/tuo-username/Cancer_classifier.git`

    3. Esegui il primo push con il token:

    `git add .`
    `git commit -m "Primo commit con token"`
    `git push origin main`



Il dataset è posizionato nella cartella data/.

[+] Cancer_classifier/
 ├── [+] data/
 │    ├── version_1.csv  → (Dataset da utilizzare)  

-Esecuzione
Per avviare il classificatore:

python main.py

Il programma chiederà:
    *Il percorso del dataset
    *Come gestire i valori mancanti
    *Come normalizzare i dati
    *Il numero di vicini k
    *Il numero di fold per la K-Fold Cross Validation

Una volta completata l’esecuzione, il modello verrà valutato e saranno generati i grafici di analisi.

-Output e Risultati

I risultati della validazione vengono salvati nella cartella:

[+] results/k-fold/
 ├── k_fold_results.csv → Contiene i risultati della validazione K-Fold

-Analisi dei risultati

Al termine dell'esecuzione, il codice genera grafici di valutazione:
    * Matrice di Confusione → Mostra il numero di predizioni corrette ed errate.
    * Curva ROC-AUC → Valuta la capacità del modello nel distinguere le classi.


-Struttura del Progetto

[+] Cancer_classifier/
 ├── [+] data_cleaning/         # Preprocessing e gestione dati
 │    ├── file_loader.py
 │    ├── data_cleaning.py
 │    ├── normalizzazione.py
 ├── [+] models/                # Modello k-NN
 │    ├── m_knn.py
 ├── [+] evaluation/             # Valutazione e grafici
 │    ├── **init**.py
 │    ├── model_evaluation.py
 │    ├── visualization.py
 ├── [+] results/                # Risultati delle validazioni
 │    ├── k-fold/
 │    │    ├── k_fold_results.csv
 ├── README.md                 # Documentazione del progetto
 ├── requirements.txt          # Dipendenze del progetto
 ├── main.py                   # Script principale