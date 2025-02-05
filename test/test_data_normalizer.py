import sys
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from data_cleaning.data_normalizer import StandardNormalizer, MinMaxNormalizer, Preprocessing


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
        pd.testing.assert_series_equal(norm_db['A'], expected)

    def test_std_normalizer(self):
        norm_db = StandardNormalizer.standardize(self.data, self.exclude_col)

        self.assertTrue(all(norm_db['C'] == self.data['C']))

        for column in norm_db.columns:
            if column not in self.exclude_col:
                self.assertAlmostEqual(norm_db[column].mean(), 0, places=6)
                self.assertAlmostEqual(norm_db[column].std(), 1, places=6)
    
    def test_invalid_method(self):
        with patch('builtins.input', return_value='invalid'):
            norm_db = Preprocessing.get_normalizer(self.data, exclude_col=self.exclude_col)
            
            # Verifica che la normalizzazione sia stata fatta con Min-Max (default)
            expected = (self.data['A'] - self.data['A'].min()) / (self.data['A'].max() - self.data['A'].min())
            pd.testing.assert_series_equal(norm_db['A'], expected)
    
    @patch('builtins.input', return_value='normalizzazione min-max')  # Simula scelta Min-Max
    def test_minmax_method(self, mock_input):
        norm_db = Preprocessing.get_normalizer(self.data, exclude_col=self.exclude_col)
        self.assertTrue(all(norm_db.columns == self.data.columns))

    @patch('builtins.input', return_value='standardizzazione')  # Simula scelta Standardizzazione
    def test_standard_method(self, mock_input):
        norm_db = Preprocessing.get_normalizer(self.data, exclude_col=self.exclude_col)
        self.assertTrue(all(norm_db.columns == self.data.columns))

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
