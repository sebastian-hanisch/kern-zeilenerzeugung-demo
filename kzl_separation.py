"""Trennungsprobleme: zu einer Aufteilung x die Koalition(en) mit dem größten Überschuss finden (Einheit je Spielklasse und Verfahren).

Alle liefern eine Liste (Maske, Überschuss), absteigend nach Überschuss; das erste Element hat den größten Überschuss über ALLE echten nichtleeren Koalitionen.

separate_knapsack          Gewichtsspiel: kleinstes x(S) unter w(S) >= q und 1 <= |S| <= n - 1 (Rucksack als MILP, scipy.optimize.milp); genau ein Treffer.
separate_facility_enum     Standortspiel: für jede Standortmenge F ist S_F = {i : x_i > d_Fi} die beste Koalition; Aufzählung aller 2^m - 1 Mengen, bis zu k Treffer.
separate_facility_milp     Standortspiel: dasselbe als gemischt-ganzzahliges Programm über (S, F); ein Treffer, Zeit 8 bis 11 ms (Plan) statt 0,1 bis 0,3 ms."""

import numpy as np

import kzl_games as G


def separate_knapsack(game, x, k: int = 1):
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = game.n
    x = np.asarray(x, dtype=float)
    res = milp(x, constraints=[LinearConstraint(game.w[None, :], game.q, np.inf), LinearConstraint(np.ones((1, n)), 1, n - 1)],
               integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)))
    if res.status == 2:    # unzulässig: keine echte Koalition erreicht die Quote (nur die große Koalition gewinnt), also kann keine verletzt sein
        return []
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"Rucksack-MILP nicht gelöst: {res.message}")
    S = res.x > 0.5
    return [(S, float(game.excess(x, S[None, :])[0]))]


def separate_facility_enum(game, x, k: int = 1):
    n = game.n
    x = np.asarray(x, dtype=float)
    margin = x[None, :] - game.D                      # (nF, n): was Kunde i bei Menge F zu x(S) beiträgt
    pos = margin > 0
    total = np.where(pos, margin, 0.0).sum(axis=1)
    cnt = pos.sum(axis=1)
    g = total - game.fk                               # bester Überschuss der Menge F (S_F = positive Beiträge)
    voll = cnt == n
    leer = cnt == 0
    if voll.any():                                    # S_F = N ist keine echte Koalition: den kleinsten Beitrag weglassen
        g = np.where(voll, total - margin.min(axis=1) - game.fk, g)
    if leer.any():                                    # S_F leer: die Einer-Koalition mit dem größten Beitrag
        g = np.where(leer, margin.max(axis=1) - game.fk, g)
    kandidaten, gesehen = [], set()
    for f in np.argsort(-g, kind="stable")[: max(4 * k, 8)]:
        S = pos[f].copy()
        if voll[f]:
            S[np.argmin(margin[f])] = False
        if leer[f]:
            S = np.zeros(n, dtype=bool)
            S[np.argmax(margin[f])] = True
        key = G.schluessel(S)
        if key in gesehen:
            continue
        gesehen.add(key)
        kandidaten.append(S)
    Sm = np.array(kandidaten)
    e = game.excess(x, Sm)
    idx = np.argsort(-e, kind="stable")[:k]
    return [(Sm[i], float(e[i])) for i in idx]


def separate_facility_milp(game, x, k: int = 1, ausgeschlossen=()):
    """Gemischt-ganzzahlig: Variablen S (n), Standortmenge y (m), Zuordnung z (n x m); ausgeschlossen: Koalitionen, die nicht wieder gefunden werden sollen."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    n, m = game.n, game.m
    x = np.asarray(x, dtype=float)
    nv = n + m + n * m
    c = np.zeros(nv)
    c[:n] = -x
    c[n:n + m] = game.fix
    c[n + m:] = game.d.ravel()
    A, lo, hi = [], [], []
    for i in range(n):
        for j in range(m):
            row = np.zeros(nv)
            row[n + m + i * m + j] = 1
            row[n + j] = -1
            A.append(row); lo.append(-np.inf); hi.append(0)
    for i in range(n):
        row = np.zeros(nv)
        row[n + m + i * m:n + m + (i + 1) * m] = 1
        row[i] = -1
        A.append(row); lo.append(0); hi.append(0)
    row = np.zeros(nv)
    row[:n] = 1
    A.append(row); lo.append(1); hi.append(n - 1)
    for Sx in ausgeschlossen:
        row = np.zeros(nv)
        row[:n] = np.where(Sx, -1.0, 1.0)
        A.append(row); lo.append(1.0 - float(np.sum(Sx))); hi.append(np.inf)
    integ = np.zeros(nv)
    integ[:n + m] = 1
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integ, bounds=Bounds(np.zeros(nv), np.ones(nv)))
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"Standort-MILP nicht gelöst: {res.message}")
    S = res.x[:n] > 0.5
    return [(S, float(game.excess(x, S[None, :])[0]))]

