def get_valid_int(prompt, min_value=1):
    """
    Chiede all'utente un numero intero valido maggiore o uguale a min_value.

    :param prompt: Messaggio da visualizzare all'utente.
    :param min_value: Valore minimo accettabile (default=1).
    :return: Numero intero valido inserito dall'utente.
    """
    while True:
        try:
            value = int(input(prompt))
            if value < min_value:
                raise ValueError(f"Il valore deve essere almeno {min_value}.")
            return value
        except ValueError as e:
            print(f"Errore: {e}. Riprova.")
