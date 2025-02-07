import numpy as np

#Questa classe si occupa della valutazione di un modello di classificazione.
#Include metriche di valutazione, validazione incrociata e strumenti di analisi visiva.
class ModelEvaluationMetrics:
    
    #Calcolo delle Metriche di Valutazione
    #Questo metodo valuta il modello restituendo metriche chiave:
    #- Accuracy
    #- Error Rate
    #- Sensibilità & Specificità
    #- Geometric Mean
    #- AUC
    @staticmethod
    def evaluate(y_true, y_pred, selected_metrics=None):
        # Definizione delle metriche valide
        valid_metrics = {"Accuracy", "Error_rate", "Sensitivity", "Specificity", "Geometric_mean", "Auc"}
        
        # Se non vengono specificate metriche, solleva un'eccezione
        if selected_metrics is None:
            raise ValueError("Errore: È necessario specificare almeno una metrica da valutare.")
        
        selected_metrics = set(selected_metrics)
        # Verifica se ci sono metriche non valide
        invalid_metrics = selected_metrics - valid_metrics
        if invalid_metrics:
            raise ValueError(f"Le seguenti metriche non sono valide: {invalid_metrics}")
        
        results = {}  # Dizionario per memorizzare i risultati delle metriche selezionate
        
        # Calcolo di Accuracy ed Error Rate solo se selezionate
        if "Accuracy" in selected_metrics or "Error_rate" in selected_metrics:
            accuracy = np.mean(y_true == y_pred)  # Accuratezza del modello
            if "Accuracy" in selected_metrics:
                results["Accuracy"] = accuracy
            if "Error_rate" in selected_metrics:
                results["Error_rate"] = 1 - accuracy
        
        # Controlla se Sensitivity, Specificity o Geometric Mean sono richieste
        if any(metric in selected_metrics for metric in ["Sensitivity", "Specificity", "Geometric_mean"]):
            sensitivities, specificities, g_means = [], [], []
            
            # Itera sulle classi presenti nei dati per calcolare le metriche necessarie
            for class_label in np.unique(y_true):
                tp = sum((y_true == class_label) & (y_pred == class_label))  # Veri positivi
                fn = sum((y_true == class_label) & (y_pred != class_label))  # Falsi negativi
                tn = sum((y_true != class_label) & (y_pred != class_label))  # Veri negativi
                fp = sum((y_true != class_label) & (y_pred == class_label))  # Falsi positivi
                
                # Calcolo di Sensitivity, Specificity e Geometric Mean
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                g_mean = np.sqrt(sens * spec) if (sens * spec) > 0 else 0
                
                sensitivities.append(sens)
                specificities.append(spec)
                g_means.append(g_mean)
            
            # Memorizza i risultati solo se la metrica è stata selezionata
            if "Sensitivity" in selected_metrics:
                results["Sensitivity"] = np.mean(sensitivities)
            if "Specificity" in selected_metrics:
                results["Specificity"] = np.mean(specificities)
            if "Geometric_mean" in selected_metrics:
                results["Geometric_mean"] = np.mean(g_means)
        
        # Calcolo dell'AUC solo se richiesto
        if "Auc" in selected_metrics:
            try:
                fpr = np.sum((y_true == 2) & (y_pred == 4)) / np.sum(y_true == 2)  # False Positive Rate
                tpr = np.sum((y_true == 4) & (y_pred == 4)) / np.sum(y_true == 4)  # True Positive Rate
                results["Auc"] = (1 + tpr - fpr) / 2  # Calcolo dell'AUC manualmente
            except ValueError as e:
                raise ValueError("Errore: Impossibile calcolare AUC.") from e
        
        return results