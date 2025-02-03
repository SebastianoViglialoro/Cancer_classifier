import numpy as np
from models.m_knn import KNNClassifier

class Modelling:
    #Classe che gestisce la creazione, l'addestramento e la predizione di modelli di machine learning.
    #Permette di astrarre il processo di training e inferenza dal tipo di modello specifico.

    def __init__(self, model_type="knn", k=3):
      
        # Inizializza il gestore del modello con il tipo di modello specificato.
        # :param model_type: Tipo di modello (attualmente supporta solo 'knn').
        # :param k: Numero di vicini da considerare nel k-NN.

        if model_type == "knn":
            self.model = KNNClassifier(k=k)
        else:
            raise ValueError("Modello non supportato. Attualmente disponibile solo k-NN.")

    def train(self, X_train, y_train):
        
        # Addestra il modello sui dati forniti.
        # :param X_train: Dati di training (feature).
        # :param y_train: Etichette corrispondenti ai dati di training.
    
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
      
        # Effettua la predizione sui dati di test.
        # :param X_test: Dati di test (feature).
        # :return: Predizioni sulle etichette.
        
        return self.model.predict(X_test)

    def get_model(self):
        
        # Restituisce il modello attuale.
        # :return: Oggetto del modello usato.
        
        return self.model
