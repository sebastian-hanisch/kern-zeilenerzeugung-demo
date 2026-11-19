"""Least-Core-LP über eine gegebene Zeilenmenge (Einheit: das LP; HiGHS über scipy).

Gewinnspiel:  min ε  unter  x(S) + ε >= v(S) für alle Zeilen S,  x(N) = v(N),  x >= 0.
Kostenspiel:  min ε  unter  x(S) - ε <= c(S) für alle Zeilen S,  x(N) = c(N),  x frei.
Die Zeilen sind echte nichtleere Koalitionen (Bool-Masken). Mit allen 2^n - 2 Zeilen ist ε das exakte ε*; mit weniger Zeilen ist ε höchstens ε*."""

from dataclasses import dataclass

import numpy as np

import kzl_games as G


@dataclass
class LpErgebnis:
    eps: float
    x: np.ndarray
    n_positiv: int          # Zeilen mit positivem Dualwert: die wirksamen Bedingungen (höchstens n + 1 bei Eckenlösung)
    positiv: np.ndarray     # Indizes dieser Zeilen


def least_core_lp(game, rows, rhs) -> LpErgebnis:
    from scipy.optimize import linprog
    n = game.n
    R = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    if game.kind == "kosten":
        A = np.hstack([R, -np.ones((len(R), 1))])
        bounds = [(None, None)] * n + [(None, None)]
        rhs_ub = b
    else:
        A = np.hstack([-R, -np.ones((len(R), 1))])
        bounds = [(0, None)] * n + [(None, None)]
        rhs_ub = -b
    res = linprog([0.0] * n + [1.0], A_ub=A, b_ub=rhs_ub, A_eq=[[1.0] * n + [0.0]], b_eq=[game.vN], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP nicht gelöst: {res.message}")
    duals = -res.ineqlin.marginals
    positiv = np.flatnonzero(duals > 1e-9)
    return LpErgebnis(float(res.x[n]), res.x[:n].copy(), int(len(positiv)), positiv)


def einer_zeilen(game):
    """Die n Einer-Koalitionen als Start der Zeilenerzeugung (für n = 1 gibt es keine echte Koalition: leere Menge)."""
    n = game.n
    if n < 2:
        return np.zeros((0, n), dtype=bool), np.zeros(0)
    S = np.eye(n, dtype=bool)
    return S, game.value_batch(S, count=False)


def full_lp(game) -> LpErgebnis:
    """Orakel: das LP mit allen 2^n - 2 echten nichtleeren Koalitionen (nur kleines n; für n = 1 gibt es nichts zu prüfen: ε* = 0)."""
    if game.n < 2:
        return LpErgebnis(0.0, np.array([game.vN]), 0, np.zeros(0, dtype=int))
    if game.n > 16:
        raise ValueError("full_lp: nur für n <= 16 (2^n Zeilen)")
    S = G.alle_koalitionen(game.n)
    return least_core_lp(game, S, game.value_batch(S, count=False))
