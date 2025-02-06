import numpy as np
import matplotlib.pyplot as plt

# Modulo di Visualizzazione (Confusion Matrix & ROC Curve)
def plot_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int) # Matrice 2x2 con rispettivamente i valori di: TN, FP, FN, TP
    cm[0, 0] = np.sum((y_true == 2) & (y_pred == 2))  # TN
    cm[0, 1] = np.sum((y_true == 2) & (y_pred == 4))  # FP
    cm[1, 0] = np.sum((y_true == 4) & (y_pred == 2))  # FN
    cm[1, 1] = np.sum((y_true == 4) & (y_pred == 4))  # TP

    # Visualizzazione matrice di confusione
    plt.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha='center', va='center', color='black')
    
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.colorbar()

    plt.savefig("result/confusion_matrix.png") #Salve l'immagine in /result


    plt.show() #Mostra la matrice di confusione

    plt.close() #Chiude la figura per evitare sovrapposizioni

def plot_auc(y_true, y_scores):
    """
    Genera e visualizza la curva ROC-AUC basata su punteggi probabilistici.
    """
    # Ordinamento delle coppie (y_scores, y_true) per FPR e TPR
    thresholds = np.sort(y_scores)[::-1]  # Ordinamento decrescente
    fpr_values = []
    tpr_values = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_scores >= threshold).astype(int)  # Classifica in base alla soglia
        fpr = np.sum((y_true == 0) & (y_pred_thresh == 1)) / np.sum(y_true == 0)
        tpr = np.sum((y_true == 1) & (y_pred_thresh == 1)) / np.sum(y_true == 1)
        
        fpr_values.append(fpr)
        tpr_values.append(tpr)

    # AUC approssimato con il metodo del trapezio
    auc_score = np.trapz(tpr_values, fpr_values)

    # Plot della Curva ROC
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")  # Linea casuale
    plt.plot(fpr_values, tpr_values, marker="o", label=f"AUC = {auc_score:.2f}")
    
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Curva ROC-AUC")
    plt.legend()
    plt.savefig("result/roc_auc_curve.png") #Salve l'immagine in /result

    plt.show() #Mostra la matrice di confusione

    plt.close() #Chiude la figura per evitare sovrapposizioni