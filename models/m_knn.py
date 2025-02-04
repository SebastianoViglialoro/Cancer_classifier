import numpy as np
import pandas as pd
import random
from collections import Counter
from abc import ABC, abstractmethod

class Classifier(ABC):
    
    # Classe astratta per un generico classificatore, così da avere più scalabilità e genericità possibile.
    # I metodi fit e predict sono metodi astratti, così da obbligare alle classi derivate da "Classifier" a
    # implementarli
    
    @abstractmethod
    def fit(self, X, y):
        #Metodo per addestrare il modello.
        pass
    
    @abstractmethod
    def predict(self, X_test):
        #Metodo per fare previsioni.
        pass

class KNNClassifier:

    #Implementazione del classificatore k-NN (k-Nearest Neighbors) da zero.
    
    def __init__(self, k=3):
        #Inizializza il modello con il numero di vicini k.
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        #Memorizza i dati di training.
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X_test):
        #Effettua una previsione su un insieme di test. Utilizza il metodo privato _predict_single() in modo iterativo
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def _predict_single(self, x):
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