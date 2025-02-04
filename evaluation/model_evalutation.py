import numpy as np
import pandas as pd
from models import KNNClassifier
from models import Modelling
from metrics_evaluation_model import ModelEvaluationMetrics
from visualization import plot_auc, plot_confusion_matrix


class Validation:
    """
    Classe che gestisce la validazione di un modello utilizzando K-Fold Cross Validation.
    """

    def __init__(self, classifier, X, y, num_folds=5, save_results=True):
        """
        Inizializza la validazione.

        :param classifier: Modello da testare.
        :param X: Feature del dataset.
        :param y: Etichette del dataset.
        :param num_folds: Numero di folds per la K-Fold Cross Validation.
        :param save_results: Se True, salva i risultati in CSV.
        """
        self.classifier = classifier
        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.num_folds = num_folds
        self.save_results = save_results

    def k_fold_cross_validation(self):
        """
        Esegue la validazione K-Fold e calcola le metriche.

        :return: DataFrame con i risultati della validazione.
        """
        fold_size = len(self.X) // self.num_folds  # Dimensione di ogni fold
        remainder = len(self.X) % self.num_folds  # Numero di elementi extra
        results = []
        all_y_true = []
        all_y_pred = []

        for i in range(self.num_folds):
            start = i * fold_size + min(i, remainder)  # Inizio del fold di test
            end = start + fold_size + (1 if i < remainder else 0)  # Fine del fold di test

            X_test, y_test = self.X[start:end], self.y[start:end]
            X_train = np.vstack((self.X[:start], self.X[end:]))  
            y_train = np.concatenate((self.y[:start], self.y[end:]))  

            self.classifier.fit(X_train, y_train)  # Addestramento del modello
            y_pred = self.classifier.predict(X_test)  # Predizione sui dati di test

            # Calcolo delle metriche (ora usiamo il metodo statico)
            metrics = ModelEvaluationMetrics.evaluate(y_test, y_pred)
            metrics["fold"] = i + 1
            results.append(metrics)

            # Salvataggio delle predizioni per i grafici
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        results_df = pd.DataFrame(results)

        # Generazione dei grafici dopo tutti i folds
        plot_confusion_matrix(np.array(all_y_true), np.array(all_y_pred))
        plot_auc(np.array(all_y_true), np.array(all_y_pred))

        # Salvataggio dei risultati
        if self.save_results:
            results_df.to_csv("../results/validation_results.csv", index=False)
            print("Risultati della validazione K-Fold salvati in results/validation_results.csv")

        return results_df
