FROM python:3.10-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia il file requirements.txt e installa le dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia il dataset dentro il container
COPY data/ /app/data/

# Copia il resto dell'applicazione
COPY . ./

# Specifica il comando di avvio
CMD ["python", "main.py"]

#docker build -t cancer_classifier .              -->Per creare l'immagine
#docker run -it --rm cancer_classifier            -->Utilizzo della modalità interattiva per gli input
#docker exec -it 'id_container' /bin/bash         -->Per entrare nel container per vedere i risultati
#docker run -it --rm cancer_classifier /bin/bash  -->Per entrare nel container
#/app/data/version_1.csv                          -->Percorso file all'interno del container docker
#docker stop $(docker ps -q)                      -->Ferma tutti i contanier
#apt update && apt install nano -y
