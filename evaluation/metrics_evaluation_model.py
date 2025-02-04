import numpy as np

# Questa classe si occupa della valutazione di un modello di classificazione.
# Include metriche di valutazione, validazione incrociata e strumenti di analisi visiva.
class ModelEvaluationMetrics:
    
    # Calcolo delle Metriche di Valutazione
    # Questo metodo valuta il modello restituendo metriche chiave:
    # - Accuracy
    # - Error Rate
    # - Sensibilità & Specificità
    # - Geometric Mean
    # - AUC
    @staticmethod
    def evaluate(y_true, y_pred):
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
            fpr = np.sum((y_true == 2) & (y_pred == 4)) / np.sum(y_true == 2)
            tpr = np.sum((y_true == 4) & (y_pred == 4)) / np.sum(y_true == 4)
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