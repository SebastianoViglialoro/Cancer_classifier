import numpy as np
import pandas as pd
import random
from collections import Counter
from abc import ABC, abstractmethod

class Classifier(ABC):
    """
    Classe astratta per un generico classificatore, così da avere più scalabilità e genericità possibile.
    I metodi fit e predict sono metodi astratti, così da obbligare alla classi derivata da "Classifier" a
    implementarli
    """
    @abstractmethod
    def fit(self, X, y):
        """Metodo per addestrare il modello."""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """Metodo per fare previsioni."""
        pass

class KNNClassifier(Classifier):
    """
    Implementazione del classificatore k-NN (k-Nearest Neighbors) da zero.
    """
    def __init__(self, k=3):
        """Inizializza il modello con il numero di vicini k."""
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Memorizza i dati di training."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X_test):
        """Effettua una previsione su un insieme di test. Utilizza il metodo privato _predict_single() in modo iterativo"""
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Effettua una previsione per un singolo punto usando la distanza euclidea."""
        # Calcola la distanza euclidea tra x e tutti i punti del training set
        distances = np.linalg.norm(self.X_train-x, axis=1) #QUI DA UN ERORRE PERCHè NON HA ANCORA I DATI PROCESSATI
        
        # Trova gli indici dei k vicini più vicini andando a prendere i primi k elementi dell'array k_indices
        k_indices = np.argsort(distances)[:self.k]
        
        # Ottiene le etichette dei k vicini
        k_nearest_labels = self.y_train[k_indices]
        
        # Determina la classe più frequente tra i vicini
        most_common = Counter(k_nearest_labels).most_common(2)
        
        # Se c'è un pareggio, sceglie casualmente tra le classi più frequenti
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            prediction = random.choice([most_common[0][0], most_common[1][0]])
        else:
            prediction = most_common[0][0]
        
        return prediction
    
if __name__ == "__main__":
    # Caricamento e preparazione del dataset
    df = pd.read_csv("data/version_1.csv")
    
    # Separazione delle feature e delle etichette
    X = df.iloc[:, :-1].values  # Tutte le colonne eccetto l'ultima
    y = df.iloc[:, -1].values   # L'ultima colonna è la classe
    
    # Suddivisione dei dati in training (80%) e test (20%)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Chiedere all'utente il valore di k
    k = int(input("Inserisci il numero di vicini (k) per il classificatore k-NN: "))
    
    # Creazione e addestramento del modello k-NN con il valore di k scelto dall'utente
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    
    # Previsione delle classi per il test set
    y_pred = knn.predict(X_test)
    
    # Calcolo dell'accuratezza del modello
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"Accuratezza del modello k-NN: {accuracy:.2f}%")