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