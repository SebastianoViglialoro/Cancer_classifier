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

    # Validazione Incrociata k-Fold del Modello
    # Suddivide il dataset in k-folds, addestra e testa il modello su ogni fold.
    # I risultati vengono aggregati e salvati in un file CSV.
    def k_fold_cross_validation(self):
        fold_size = len(self.X) // self.num_folds #Dimensione di ogni fold
        remainder = len(self.X) % self.num_folds #Numero di elementi extra
        results = []

        for i in range(self.num_folds):
            start = i * fold_size + min(i, remainder) #Determina l'inizio del fold di test
            end = start + fold_size + (1 if i < remainder else 0) #Determina la fine del fold di test

            X_test, y_test = self.X[start:end], self.y[start:end]
            X_train = np.vstack((self.X[:start], self.X[end:])) #Concatenazione campioni prima e dopo il fold di test
            y_train = np.concatenate((self.y[:start], self.y[end:])) #Stessa cosa di X_train solo che essendo ad una dimensione usiamo concatenate()
            
            self.classifier.fit(X_train, y_train) #Si addestra il modello sui dati di training e sulle relative etichette
            y_pred = self.classifier.predict(X_test) #Una volta addestrato si classificano i nuovi dati e si generano le previsioni delle etichette per i dati di test
            
            metrics = self.evaluate(y_test, y_pred)
            metrics["fold"] = i + 1 #Questo viene messo all'inizio automaticamente da pandas perchè viene aggiunto separatamente
            results.append(metrics)
        
        results_df = pd.DataFrame(results)

        if self.save_results: #Se save_result è true allora salva i risultati in un file CSV
            results_df.to_csv("../results/validation_results.csv", index=False)
            print("Risultati della validazione k-Fold salvati in results/validation_results.csv")
        
        return results_df