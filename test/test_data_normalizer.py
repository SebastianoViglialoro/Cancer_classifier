import unittest
import pandas as pd
import numpy as np
from data_cleaning import Preprocessing, MinMaxNormalizer, StandardNormalizer

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Dataset di esempio per i test
        self.data = pd.DataFrame({
            "A": [10, 20, 30, 40],
            "B": [1.0, 2.0, 3.0, 4.0],
            "C": [100, 200, 300, 400]
        })
        self.exclude_col = ['C']
    
    def test_min_max_normalizer(self):
        norm_db = MinMaxNormalizer.normalize(self.data, self.exclude_col)

        # Controllo che le colonne escluse siano rimaste invariate
        self.assertTrue(all(norm_db['C'] == self.data['C']))

        # Controllo che le colonne normalizzate siano nel range [0, 1]
        for column in norm_db.columns:
            if column not in self.exclude_col:
                self.assertTrue(all(norm_db[column] >= 0) and all(norm_db[column] <= 1))
        
        #verifico il risultato atteso per una colonna
        expected = (self.data['A'] - self.data['A'].min()) / (self.data['A'].max() - self.data['A'].min())
        pd.testing.assert_series_equal(all(norm_db['A'], expected))

    def test_std_normalizer(self):
        norm_db = StandardNormalizer.normalize(self.data, self.exclude_col)

        # Controllo che le colonne escluse siano rimaste invariate
        self.assertTrue(all(norm_db['C'] == self.data['C']))

        # Controllo che le colonne normalizzate siano nel range [0, 1]
        for column in norm_db.columns:
            if column not in self.exclude_col:
                self.assertTrue(all(norm_db[column] >= 0) and all(norm_db[column] <= 1))
        
        #verifico il risultato atteso per una colonna
        expected = (self.data['A'] - self.data['A'].min()) / (self.data['A'].max() - self.data['A'].min())
        pd.testing.assert_series_equal(all(norm_db['A'], expected))
    
    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            Preprocessing.get_normalizer(self.data, exclude_col=self.exclude_col, method='invalid')
    
    def test_minmax_method(self):
        norm_db = Preprocessing.get_normalizer(self.data, exclude_col=self.exclude_col, method='minmax')
        self.assertTrue(all(norm_db.columns == self.data.columns))

if __name__ == "__main__":
    unittest.main()