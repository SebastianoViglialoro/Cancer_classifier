import numpy as np
import pandas as pd
from models import Modelling
from evaluation.metrics_evaluation_model import ModelEvaluationMetrics
from evaluation.visualization import plot_auc, plot_confusion_matrix


class Validation:
    
    #Classe che gestisce la validazione di un modello utilizzando K-Fold Cross Validation.
    
    def __init__(self, classifier, X, y, num_folds=5, save_results=True):
        
        #Inizializza la validazione.

        #:param classifier: Modello da testare.
        #:param X: Feature del dataset.
        #:param y: Etichette del dataset.
        #:param num_folds: Numero di folds per la K-Fold Cross Validation.
        #:param save_results: Se True, salva i risultati in CSV.
        
        self.classifier = classifier
        self.X = np.array(X, dtype=float) #Convertiamo le feature in float per sicurezza
        self.y = np.array(y)
        self.num_folds = num_folds
        self.save_results = save_results

    def k_fold_cross_validation(self):
        
        #Esegue la validazione K-Fold e calcola le metriche.

        #:return: DataFrame con i risultati della validazione.
        
        fold_size = len(self.X) // self.num_folds  #Dimensione di ogni fold
        remainder = len(self.X) % self.num_folds  #Numero di elementi extra
        results = []
        all_y_true = []
        all_y_pred = []
        all_y_scores = []  #Ora memorizziamo le probabilità invece delle classi predette

        for i in range(self.num_folds):
            start = i * fold_size + min(i, remainder)  #Inizio del fold di test
            end = start + fold_size + (1 if i < remainder else 0)  #Fine del fold di test

            X_test, y_test = self.X[start:end], self.y[start:end]
            X_train = np.vstack((self.X[:start], self.X[end:])) #Dati di addestramento
            y_train = np.concatenate((self.y[:start], self.y[end:])) #Etichette di addestramento

            self.classifier.train(X_train, y_train)  #Addestramento del modello
            y_pred, y_scores = self.classifier.predict(X_test)  #Ora restituiamo anche le probabilità

            #Calcolo delle metriche
            metrics = ModelEvaluationMetrics.evaluate(y_test, y_pred)
            metrics["fold"] = i + 1
            results.append(metrics)

            #Salvataggio delle predizioni per i grafici
            all_y_pred.extend(y_pred)
            all_y_true.extend(y_test)
            y_true_binary = np.where(np.array(all_y_true) == 4, 1, 0) #Conversione per la ROC-AUC
            all_y_scores.extend(y_scores)  #Usiamo le probabilità per la ROC-AUC

        results_df = pd.DataFrame(results)

        #Cacolo matrice di confusione
        plot_confusion_matrix(np.array(all_y_true), np.array(all_y_pred))  #Usa classi discrete

        #Calcolo della ROC-AUC basata sulle probabilità
        plot_auc(np.array(y_true_binary), np.array(all_y_scores))  #ROC-AUC basata sulle probabilità

        #Salvataggio dei risultati
        if self.save_results:
            results_df.to_csv("result/validation_results.csv", index=False)
            print("Risultati della validazione K-Fold salvati in results/validation_results.csv")

        return results_df