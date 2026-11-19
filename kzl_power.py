"""Shapley-Shubik-Index im Gewichtsspiel als Aufteilung (Brücke zu `shapley-demo` und zum Machtindex), exakt per Zähl-DP statt über n! Reihenfolgen.

phi_i = Summe über Koalitionen S ohne i mit w(S) < q <= w(S) + w_i von |S|! (n - |S| - 1)! / n!. Die Zahl der Teilmengen nach (Größe, Gewichtssumme) wird einmal
für alle Spieler gezählt (exakte ganze Zahlen) und für jeden Spieler i durch Rückrechnung ohne i gewonnen; die Antwort ist exakt (Bruch, als float ausgegeben)."""

from math import factorial

import numpy as np

MAX_ELEMENTE = 3_000_000   # n * (Gewichtssumme + 1): größer wird die Zähltabelle zu groß für die Live-Rechnung


def shapley_shubik(w, q) -> np.ndarray:
    w = np.asarray(w).astype(int)
    n = len(w)
    W = int(w.sum())
    if n * (W + 1) > MAX_ELEMENTE:
        raise ValueError(f"Gewichtssumme {W} bei n = {n} zu groß für die Zähltabelle")
    q = int(round(q))
    zaehl = np.zeros((n + 1, W + 1), dtype=object)
    zaehl[:] = 0
    zaehl[0, 0] = 1
    for wj in w:
        neu = zaehl.copy()
        neu[1:, wj:] = neu[1:, wj:] + zaehl[:-1, :W + 1 - wj]
        zaehl = neu
    fakt = [factorial(k) for k in range(n + 1)]
    phi = np.zeros(n)
    for i, wi in enumerate(w):
        ohne = zaehl.copy()                     # Teilmengen ohne i, nach (Größe k, Summe s)
        for k in range(1, n + 1):
            ohne[k, wi:] = zaehl[k, wi:] - ohne[k - 1, :W + 1 - wi]
        lo, hi = max(0, q - wi), q - 1
        zaehler = 0
        if hi >= lo:
            for k in range(0, n):
                zaehler += fakt[k] * fakt[n - k - 1] * int(ohne[k, lo:hi + 1].sum())
        phi[i] = zaehler / fakt[n]
    return phi
