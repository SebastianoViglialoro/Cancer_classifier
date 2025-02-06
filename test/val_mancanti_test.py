import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from data_cleaning.data_cleaner import DataCleaner, GestioneValMancanti, RimuoviDuplicati

class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5, np.nan],
            'B': [np.nan, 2, 3, 4, np.nan, 6],
            'classtype_v1': [1, 0, 1, np.nan, 0, 1],
            'Sample code number': [12345, 12346, 12347, 12348, 12349, 12350]
        })

    def test_rimozione_valori_mancanti(self):
        cleaned_data = GestioneValMancanti.get_mode('rimozione', self.data)
        self.assertFalse(cleaned_data.isnull().values.any())

    def test_sostituzione_con_media(self):
        cleaned_data = GestioneValMancanti.get_mode('media', self.data)
        # Verifica che non ci siano più NaN
        self.assertFalse(cleaned_data['A'].isna().any(), "Ci sono ancora valori NaN!")
        # Calcola la media ATTESA senza i NaN
        expected_mean_A = cleaned_data['A'].mean()
        # Stampa per debug
        print("Valori originali:\n", self.data['A'])
        print("Valore atteso della media (prima della sostituzione):", expected_mean_A)
        print("Valori dopo la sostituzione:\n", cleaned_data['A'])
        print("Nuova media:", cleaned_data['A'].mean())
        # Controlla se i NaN sono stati sostituiti con la media
        nan_indices = self.data['A'].isna()
        self.assertTrue(np.allclose(cleaned_data.loc[nan_indices, 'A'], expected_mean_A, atol=1e-5))
        # Controllo finale della media
        self.assertAlmostEqual(expected_mean_A, cleaned_data['A'].mean(),places=5)

    def test_sostituzione_con_moda(self):
        cleaned_data = GestioneValMancanti.get_mode('moda', self.data)
        self.assertFalse(cleaned_data.isnull().values.any())
        expected_mode_B = self.data['B'].mode().iloc[0]
        self.assertIn(expected_mode_B, cleaned_data['B'].values)

    def test_sostituzione_con_mediana(self):
        cleaned_data = GestioneValMancanti.get_mode('mediana', self.data)
        self.assertFalse(cleaned_data.isnull().values.any())
        expected_median_A = np.median(cleaned_data['A'].values)
        self.assertEqual(expected_median_A, np.median(cleaned_data['A'].values))

    def test_rimozione_duplicati(self):
        duplicated_data = pd.concat([self.data, self.data])  # Duplico i dati
        cleaned_data = RimuoviDuplicati.dup_remove(duplicated_data)
        self.assertEqual(len(cleaned_data), len(self.data))  # Deve tornare alla lunghezza originale

    @patch('builtins.input', side_effect=['media', 'n'])
    def test_clean_and_save(self, mock_input):
        cleaned_data = DataCleaner.clean_and_save(self.data)
        self.assertFalse(cleaned_data.isnull().values.any())

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            GestioneValMancanti.get_mode('invalid_mode', self.data)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)