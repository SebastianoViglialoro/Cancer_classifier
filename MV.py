import numpy as np
import pandas as pd
import random
from collections import Counter
from abc import ABC, abstractmethod
from evaluation.visualization import plot_confusion_matrix, plot_auc 
#qua non si dovrebbe scrivere evaluation.visualization ma solo .visualization

"""
DOVREBBE RISOLVERE L'ISSUE DEL PROF, è DA CONTROLLARE!!
"""

class Classifier(ABC):
    """
    Classe astratta per un generico classificatore.
    """
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass

class KNNClassifier(Classifier):
    """
    Implementazione del classificatore k-NN.
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        return np.array([self._predict_single(x) for x in X_test])
    
    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(2)
        return random.choice([most_common[0][0], most_common[1][0]]) if len(most_common) > 1 and most_common[0][1] == most_common[1][1] else most_common[0][0]

class MV:
    """
    Classe che implementa sia il Modelling che la Validation.
    """
    def __init__(self, classifier, X, y, num_folds=5, save_results=True):
        self.classifier = classifier
        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.num_folds = num_folds
        self.save_results = save_results
    
    def evaluate(self, y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        error_rate = 1 - accuracy
        sensitivities, specificities, g_means = [], [], []
        
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
        
        sensitivity = np.mean(sensitivities)
        specificity = np.mean(specificities)
        geometric_mean = np.mean(g_means)
        
        try:
            fpr = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
            tpr = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
            auc = (1 + tpr - fpr) / 2
        except ValueError:
            auc = None
        
        return {"accuracy": accuracy, "error_rate": error_rate, "sensitivity": sensitivity, "specificity": specificity, "geometric_mean": geometric_mean, "auc": auc}
    
    def k_fold_cross_validation(self):
        fold_size = len(self.X) // self.num_folds
        remainder = len(self.X) % self.num_folds
        results, all_y_true, all_y_pred = [], [], []
        
        for i in range(self.num_folds):
            start = i * fold_size + min(i, remainder)
            end = start + fold_size + (1 if i < remainder else 0)
            
            X_test, y_test = self.X[start:end], self.y[start:end]
            X_train = np.vstack((self.X[:start], self.X[end:]))
            y_train = np.concatenate((self.y[:start], self.y[end:]))
            
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            
            metrics = self.evaluate(y_test, y_pred)
            metrics["fold"] = i + 1
            results.append(metrics)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        
        results_df = pd.DataFrame(results)
        plot_confusion_matrix(np.array(all_y_true), np.array(all_y_pred))
        plot_auc(np.array(all_y_true), np.array(all_y_pred))
        
        if self.save_results:
            results_df.to_csv("../results/validation_results.csv", index=False)
            print("Risultati salvati in results/validation_results.csv")
        
        return results_df
