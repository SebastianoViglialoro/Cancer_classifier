import numpy as np
import pandas as pd
import random
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, num_folds=5):
        """
        Inizializza il classificatore KNN.
        
        :param k: Numero di vicini da considerare.
        :param num_folds: Numero di fold per la validazione incrociata.
        """
        self.k = k
        self.num_folds = num_folds

    @staticmethod
    def euclidean_distance(x1, x2):
        """
        Calcola la distanza euclidea tra due punti.
        
        :param x1: Primo punto.
        :param x2: Secondo punto.
        :return: Distanza euclidea.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def classify(self, X_train, y_train, X_test):
            """
            Classifica i punti di test basandosi sui k vicini più vicini.
            
            :param X_train: Dati di addestramento.
            :param y_train: Etichette di addestramento.
            :param X_test: Dati di test.
            :return: Array con le predizioni.
            """
            predictions = []
            
            for test_point in X_test:
                # Calcolo delle distanze tra il punto di test e tutti i punti di training
                distances = [self.euclidean_distance(test_point, x) for x in X_train]
                
                # Ottieni gli indici dei k punti più vicini
                k_indices = np.argsort(distances)[:self.k]
                
                # Ottieni le etichette dei k vicini
                k_nearest_labels = [y_train[i] for i in k_indices]
                
                # Conta le occorrenze di ogni classe tra i vicini
                most_common = Counter(k_nearest_labels).most_common()
                
                # Se c'è un pareggio, sceglie casualmente tra le classi più frequenti
                if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                    prediction = random.choice([most_common[0][0], most_common[1][0]])
                else:
                    prediction = most_common[0][0]
                
                predictions.append(prediction)
            
            return np.array(predictions)
    
    
    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Calcola accuratezza, sensibilità e specificità del modello.
        
        :param y_true: Etichette vere.
        :param y_pred: Etichette predette.
        :return: Accuratezza, sensibilità e specificità.
        """
        accuracy = np.mean(y_true == y_pred)
        tp = sum((y_true == 1) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        tn = sum((y_true == 0) & (y_pred == 0))
        fp = sum((y_true == 0) & (y_pred == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return accuracy, sensitivity, specificity
            
    def k_fold_cross_validation(self, X, y):
        """
        Esegue la validazione incrociata k-fold.
        
        :param X: Dati di input.
        :param y: Etichette.
        :return: Media di accuratezza, sensibilità e specificità.
        """
        fold_size = len(X) // self.num_folds
        accuracies, sensitivities, specificities = [], [], []
        
        for i in range(self.num_folds):
            start, end = i * fold_size, (i + 1) * fold_size
            X_test, y_test = X[start:end], y[start:end]
            X_train = np.vstack((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            
            y_pred = self.classify(X_train, y_train, X_test)
            accuracy, sensitivity, specificity = self.evaluate(y_test, y_pred)
            
            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        return np.mean(accuracies), np.mean(sensitivities), np.mean(specificities)