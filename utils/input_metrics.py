def select_metrics():
    print("\nSeleziona le metriche da valutare (inserisci i numeri separati da virgola):")
    print("1 - Accuracy Rate")
    print("2 - Error Rate")
    print("3 - Sensitivity")
    print("4 - Specificity")
    print("5 - Geometric Mean")
    print("6 - Area Under the Curve")
    print("7 - Tutte le metriche")

    metric_options = {
        "1": "Accuracy",
        "2": "Error_rate",
        "3": "Sensitivity",
        "4": "Specificity",
        "5": "Geometric_mean",
        "6": "Auc"
    }

    while True:  # Continua a chiedere finché l'input non è valido
        choices = input("Scelta: ").split(",")
        selected_metrics = []
        invalid_entries = []

        # Controlla ogni valore inserito
        for choice in choices:
            choice = choice.strip()

            if choice == "7":  # Se "7" è selezionato, verifica che non ci siano errori
                if invalid_entries:
                    print(f"Errore: Hai inserito valori non validi {invalid_entries}. Riprova.")
                    break
                return list(metric_options.values())  # Restituisce tutte le metriche

            if choice not in metric_options:
                invalid_entries.append(choice)
            else:
                selected_metrics.append(metric_options[choice])

        # Se ci sono errori, chiedi di riprovare
        if invalid_entries:
            print(f"Errore: '{', '.join(invalid_entries)}' non sono scelte valide. Inserisci solo numeri da 1 a 7.")
            print("Riprova inserendo una selezione valida.")
        else:
            return selected_metrics  # Restituisce solo metriche valide
