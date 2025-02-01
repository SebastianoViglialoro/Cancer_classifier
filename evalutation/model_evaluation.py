import numpy as np

class ModelEvaluation:
    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Calcola accuratezza, sensibilità e specificità del modello.
        
        :param y_true: Etichette vere.
        :param y_pred: Etichette predette.
        :return: Accuratezza, sensibilità e specificità.
        """
        accuracy = np.mean(y_true == y_pred)
        tp = sum((y_true == 1) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        tn = sum((y_true == 0) & (y_pred == 0))
        fp = sum((y_true == 0) & (y_pred == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return accuracy, sensitivity, specificity

    def k_fold_cross_validation(self, classifier, X, y, num_folds=5):
        """
        Esegue la validazione incrociata k-fold.
        
        :param classifier: Modello di classificazione da testare.
        :param X: Dati di input.
        :param y: Etichette.
        :param num_folds: Numero di fold per la validazione incrociata.
        :return: Media di accuratezza, sensibilità e specificità.
        
        """
        fold_size = len(X) // num_folds
        accuracies, sensitivities, specificities = [], [], []
        
        for i in range(num_folds):
            start, end = i * fold_size, (i + 1) * fold_size
            X_test, y_test = X[start:end], y[start:end]
            X_train = np.vstack((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            accuracy, sensitivity, specificity = self.evaluate(y_test, y_pred)
            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        return np.mean(accuracies), np.mean(sensitivities), np.mean(specificities)