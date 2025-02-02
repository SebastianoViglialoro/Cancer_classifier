import numpy as np
import matplotlib.pyplot as plt

#Modulo di Visualizzazione (Confusion Matrix & ROC Curve)
def plot_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int) #Matrice 2x2 con rispettivamente i valori di: TN, FP, FN, TP
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # TN
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # FP
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # FN
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # TP

    #Visualizzazione matrice di confusione
    plt.figure(figsize=(5,5))
    plt.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha='center', va='center', color='black')
    
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.colorbar()
    plt.show()

def plot_auc(y_true, y_pred):
    #Genera e visualizza la curva ROC e calcola l'AUC.
    try:
        fpr = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0) #False Negative Rate
        tpr = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1) #False Positive Rate
        auc_score = (1 + tpr - fpr) / 2
        
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random') #Modello casuale. Se il nostro di avvicina alla linea tratteggiata non è un buon modello
        plt.plot([0, 1], [0, auc_score], marker='o', label=f'AUC = {auc_score:.2f}')
        
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Curva ROC-AUC")
        plt.legend()
        plt.show()
    
    except ValueError:
        print("AUC non calcolabile per una sola classe.")