import numpy as np
import pandas as pd
import random
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, num_folds=5):
        self.k = k
        self.num_folds = num_folds

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def classify(self, X_train, y_train, X_test):
            predictions = []
            
            for test_point in X_test:
                distances = [self.euclidean_distance(test_point, x) for x in X_train]
                k_indices = np.argsort(distances)[:self.k]
                k_nearest_labels = [y_train[i] for i in k_indices]
                
                most_common = Counter(k_nearest_labels).most_common()
                if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                    prediction = random.choice([most_common[0][0], most_common[1][0]])
                else:
                    prediction = most_common[0][0]
                
                predictions.append(prediction)
            
            return np.array(predictions)
    
    
    @staticmethod
    def evaluate(y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        tp = sum((y_true == 1) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        tn = sum((y_true == 0) & (y_pred == 0))
        fp = sum((y_true == 0) & (y_pred == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return accuracy, sensitivity, specificity
            
    def k_fold_cross_validation(self, X, y):
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