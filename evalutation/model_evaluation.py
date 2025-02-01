import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Questa classe si occupa della valutazione di un modello di classificazione.
# Include metriche di valutazione, validazione incrociata e strumenti di analisi visiva.
class ModelEvaluation:
    
    # Inizializzazione della Classe
    # Configura il classificatore, i dati e il numero di folds per la validazione
    def _init_(self, classifier, X, y, num_folds=5, save_results=True):
        self.classifier = classifier
        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.num_folds = num_folds
        self.save_results = save_results


    # Calcolo delle Metriche di Valutazione
    # Questo metodo valuta il modello restituendo metriche chiave:
    # - Accuracy
    # - Error Rate
    # - Sensibilità & Specificità
    # - Geometric Mean
    # - AUC
    def evaluate(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred) #Calcolo accuratezza modello
        error_rate = 1 - accuracy
        sensitivities, specificities, g_means = [], [], []
        auc = None

        # Calcola metriche per ogni classe nel dataset:
        for class_label in np.unique(y_true):
            tp = sum((y_true == class_label) & (y_pred == class_label))
            fn = sum((y_true == class_label) & (y_pred != class_label))
            tn = sum((y_true != class_label) & (y_pred != class_label))
            fp = sum((y_true != class_label) & (y_pred == class_label))
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            g_mean = np.sqrt(sens * spec) if (sens * spec) > 0 else 0
            
            sensitivities.append(sens)
            specificities.append(spec)
            g_means.append(g_mean)
        
        # Aggregazione delle metriche per avere un'unica misura rappresentativa
        sensitivity = np.mean(sensitivities)
        specificity = np.mean(specificities)
        geometric_mean = np.mean(g_means)

        # Calcolo manuale dell'AUC (Area Under Curve)
        try:
            fpr = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
            tpr = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
            auc = (1 + tpr - fpr) / 2
        except ValueError as e:
            raise ValueError("Errore: Impossibile calcolare AUC. Verificare i dati di input.") from e

        return {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "geometric_mean": geometric_mean,
            "auc": auc
        }

    def k_fold_cross_validation(self, classifier, X, y, num_folds=5):
        """
        Esegue la validazione incrociata k-fold.
        
        :param classifier: Modello di classificazione da testare.
        :param X: Dati di input.
        :param y: Etichette.
        :param num_folds: Numero di fold per la validazione incrociata.
        :return: Media di accuratezza, sensibilità e specificità.
        
        """
        fold_size = len(X) // num_folds
        accuracies, sensitivities, specificities = [], [], []
        
        for i in range(num_folds):
            start, end = i * fold_size, (i + 1) * fold_size
            X_test, y_test = X[start:end], y[start:end]
            X_train = np.vstack((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            accuracy, sensitivity, specificity = self.evaluate(y_test, y_pred)
            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        return np.mean(accuracies), np.mean(sensitivities), np.mean(specificities)