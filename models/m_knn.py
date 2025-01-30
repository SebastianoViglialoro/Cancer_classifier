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