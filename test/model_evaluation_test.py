import unittest
import numpy as np
from models import Modelling
from evaluation import Validation
from evaluation import ModelEvaluationMetrics

class TestModelValidation(unittest.TestCase):
    """Test per la validazione K-Fold del modello"""

    def setUp(self):
        """Inizializza dati di test prima di ogni test"""
        #Creiamo un dataset grande per una valutazione più accurata
        self.X_test = np.array([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
            [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
            [16, 17], [17, 18], [18, 19], [19, 20], [20, 21]
        ])

        #Etichette bilanciate tra 2 e 4
        self.y_test = np.array([
            2, 2, 4, 4, 2, 
            4, 2, 2, 4, 4, 
            2, 2, 4, 4, 2, 
            4, 2, 2, 4, 4
        ])


    def test_k_fold_execution(self):
        """Testa l'intero processo di validazione K-Fold con output probabilistici"""
        print("\nDEBUG - Dataset di test:")
        print("Feature (X_test) - Prime 5 righe:\n", self.X_test[:5])
        print("Labels (y_test) - Prime 5 etichette:\n", self.y_test[:5])
        print("Classi uniche nei dati di test:", np.unique(self.y_test))

        #Creiamo l'istanza di Modelling invece di KNNClassifier direttamente
        model_manager = Modelling(model_type="knn", k=3)  # Usa il modello k-NN

        #Inizializziamo la classe di validazione con il modello corretto
        validator = Validation(classifier=model_manager, X=self.X_test, y=self.y_test, num_folds=3)

        print("\nDEBUG - Dimensioni del training set in K-Fold:")
        print("X_train shape:", self.X_test.shape, "y_train shape:", self.y_test.shape)

        #Eseguiamo la validazione K-Fold PRIMA di chiamare predict**
        results_df = validator.k_fold_cross_validation()

        #Ora il modello è addestrato, possiamo chiamare predict**
        print("\nDEBUG - Predizioni nel test K-Fold:")
        all_y_pred, all_y_scores = validator.classifier.predict(self.X_test)
        print("Predicted labels:", all_y_pred)
        print("Predicted scores:", all_y_scores)
        print("Classi uniche predette:", np.unique(all_y_pred))

        #Verifica che i risultati siano stati prodotti
        self.assertFalse(results_df.empty, "La validazione K-Fold non ha prodotto risultati.")

        #Verifica che il DataFrame abbia le metriche attese
        expected_columns = {"fold", "accuracy", "error_rate", "sensitivity", "specificity", "geometric_mean", "auc"}
        # Normalizza i nomi delle colonne in lowercase per evitare problemi di capitalizzazione
        df_columns = set(map(str.lower, results_df.columns))
        expected_columns_lower = set(map(str.lower, expected_columns))

        self.assertTrue(expected_columns_lower.issubset(df_columns),
                f"Le colonne dei risultati non contengono tutte le metriche attese: {results_df.columns}")

        #Verifica che le probabilità siano nel range corretto
        self.assertTrue(all(0 <= p <= 1 for p in all_y_scores),
                        f"Le probabilità predette non sono nel range corretto: {all_y_scores}")
        
    def test_metric_calculations(self):
        """Verifica il calcolo delle metriche di valutazione"""
        y_true = np.array([2, 2, 4, 4, 2])
        y_pred = np.array([2, 2, 4, 4, 4])  # Un errore intenzionale per verificare il comportamento
        metrics = ModelEvaluationMetrics.evaluate(y_true, y_pred)

        # Convertiamo tutte le chiavi in lowercase per uniformità
        metrics_lower = {k.lower(): v for k, v in metrics.items()}

        self.assertIn("accuracy", metrics_lower)
        self.assertIn("auc", metrics_lower)
        self.assertTrue(0 <= metrics_lower["auc"] <= 1)  # AUC deve essere tra 0 e 1



if __name__ == '__main__':
    unittest.main()
