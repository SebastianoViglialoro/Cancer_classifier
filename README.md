# Cancer Classifier

## Descrizione
La classificazione delle cellule tumorali è essenziale per una diagnosi accurata e tempestiva. Esse si suddividono in due categorie principali:
- **Benigne**: rimangono circoscritte e non invadono altri tessuti.
- **Maligne**: caratterizzate da una crescita incontrollata con capacità di metastatizzare.

L’analisi morfologica e biologica di queste cellule, basata su parametri come forma, dimensioni e struttura nucleare, consente di distinguerle e supportare le decisioni cliniche.

Questo progetto mira a sviluppare un modello di **Intelligenza Artificiale** per classificare le cellule tumorali utilizzando l'algoritmo **k-Nearest Neighbors (k-NN)**. Il dataset è sottoposto a processi di pulizia e normalizzazione per migliorare l'affidabilità delle previsioni. Per garantire una valutazione solida del modello, viene utilizzata la **validazione incrociata K-Fold**, permettendo una stima più accurata delle prestazioni.

Il sistema è altamente personalizzabile, offrendo agli utenti la possibilità di scegliere:
- Il metodo di gestione dei valori mancanti.
- La tecnica di normalizzazione.
- Il numero di vicini **k**.
- Il numero di fold per la validazione K-Fold.

Al termine dell’analisi, il modello produce **report dettagliati** e **rappresentazioni grafiche**, tra cui la **matrice di confusione** e la **curva ROC-AUC**, strumenti fondamentali per valutare l’efficacia della classificazione e il potenziale utilizzo in ambito medico.

---
## Approccio adottato
Il progetto è stato sviluppato seguendo un'architettura **modulare e scalabile**, con il codice suddiviso in diverse componenti:

### **1. Preprocessing dei dati**
- Gestione dei valori mancanti.
- Normalizzazione dei dati per garantire uniformità.

### **2. Modellizzazione**
- Implementazione dell'algoritmo **k-NN** **senza l’uso di Scikit-Learn**, per una comprensione approfondita del funzionamento.

### **3. Validazione**
- Utilizzo della **K-Fold Cross Validation** per una valutazione robusta delle prestazioni del modello.

### **4. Visualizzazione dei risultati**
- Generazione automatica di **grafici di analisi**, tra cui:
  - **Matrice di confusione** → Per valutare la precisione del modello.
  - **Curva ROC-AUC** → Per analizzare la capacità di distinzione tra classi.

Abbiamo evitato l’uso di librerie preconfezionate per il machine learning, implementando manualmente i metodi di classificazione e validazione per comprendere meglio ogni fase del processo.

---
## Installazione
### 1️ Clona il repository
```sh
git clone https://github.com/tuo-username/Cancer_classifier.git
cd Cancer_classifier
```

### 2️ Installa le dipendenze
```sh
pip install -r requirements.txt
```

### 3️ Assicurati che il dataset sia presente
Il dataset deve trovarsi nella cartella **data/**:
```
[+] Cancer_classifier/
│── [+] data/
│   ├── version_1.csv → (Dataset da utilizzare)
```

### Installazione su Mac
1. **Genera un token di accesso personale (PAT):**  
   Vai su **GitHub → Settings → Developer settings → Personal access tokens** e crea un nuovo token con permessi per il repository.

2. **Configura Git con il token:**
```sh
git config --global user.name "IlTuoNomeUtente"
git config --global user.email "laTuaEmail@esempio.com"
git remote set-url origin https://<TOKEN>@github.com/tuo-username/Cancer_classifier.git
```

3. **Esegui il primo push con il token:**
```sh
git add .
git commit -m "Primo commit con token"
git push origin main
```

---
## Esecuzione
Per avviare il classificatore, eseguire:
```sh
python main.py
```
Il programma richiederà di specificare:
-  **Percorso del dataset**
-  **Gestione dei valori mancanti**
-  **Metodo di normalizzazione**
-  **Numero di vicini k**
-  **Numero di fold per la validazione K-Fold**

Al termine dell’esecuzione, verranno prodotti **risultati dettagliati** e **grafici di valutazione**.

---
##  Output e Risultati
I risultati della validazione vengono salvati nella cartella:
```
[+] results/k-fold/
│── k_fold_results.csv → Contiene i risultati della validazione K-Fold
```

Al termine dell'esecuzione, il codice genera **grafici di valutazione**:
- **Matrice di Confusione** → Mostra il numero di predizioni corrette ed errate.
- **Curva ROC-AUC** → Valuta la capacità del modello nel distinguere le classi.

---
## Struttura del Progetto
```
[+] Cancer_classifier/
│── [+] data/             # Importazione del CSV
│   ├── version_1.csv

│── [+] data_cleaning/    # Preprocessing e gestione dati
│   ├── __init__.py
│   ├── data_cleaner.py
│   ├── data_normalizer.py
│   ├── file_importer.py

│── [+] evaluation/       # Valutazione e grafici
│   ├── __init__.py
│   ├── metrics_evaluation.py
│   ├── model_evaluation.py
│   ├── visualization.py

│── [+] models/           # Modello k-NN
│   ├── __init__.py
│   ├── m_knn.py
│   ├── model_management.py

│── [+] utils/ 
│   ├── input_valid_int.py

│── [+] results/          # Risultati delle validazioni
│   ├── k-fold/
│   │   ├── k_fold_results.csv

│── README.md             # Documentazione del progetto
│── requirements.txt      # Dipendenze del progetto
│── project_setup.py      # Task affiliate al gruppo 8
│── main.py               # Script principale
```

---
## Conclusioni
Questo progetto rappresenta un passo importante verso l'integrazione dell'Intelligenza Artificiale nell’analisi delle cellule tumorali. Grazie a un'implementazione manuale del k-NN, alla validazione rigorosa e agli strumenti di visualizzazione, il modello offre una base solida per applicazioni mediche e di ricerca.

Per miglioramenti futuri, si potrebbe valutare l'integrazione di algoritmi più avanzati come **SVM, Random Forest o reti neurali**.

**Per domande o suggerimenti:** contattaci via GitHub!

