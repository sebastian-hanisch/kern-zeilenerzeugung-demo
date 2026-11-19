"""Zeilenerzeugung für den Least Core: mit wenigen Bedingungen starten, die am stärksten verletzte nachlegen (Einheit: das Verfahren, ohne eigene Regel).

Schleife: (1) LP über die bisherigen Zeilen lösen -> (ε, x); (2) Trennung: die Koalitionen mit dem größten Überschuss unter x suchen;
(3) übertrifft ein Überschuss ε um mehr als die Schwelle, kommt die Koalition als neue Zeile hinzu, sonst Ende: x erfüllt ALLE 2^n - 2 Bedingungen
bis auf die Schwelle, ε ist ε*. Eine feste Obergrenze der Schleifen gibt es nicht; wer eine setzt (`max_schleifen`), bekommt `abgebrochen=True` und das Ergebnis
ist dann nur eine untere Schranke von ε*."""

import time
from dataclasses import dataclass, field

import numpy as np

import kzl_games as G
import kzl_lp as LP

REL_TOL = 1e-7   # Schwelle relativ zu Wert(N): eine Koalition zählt als verletzt, wenn ihr Überschuss ε um mehr als REL_TOL * max(1, Wert(N)) übertrifft


@dataclass
class SchleifenSchritt:
    zeilen_vorher: int
    eps: float                       # ε des LP in dieser Schleife (Ergebnis vor der Trennung)
    schnitte: list = field(default_factory=list)   # [(Maske, Überschuss unter dem x dieser Schleife)] der neu aufgenommenen Zeilen


@dataclass
class RowgenErgebnis:
    eps: float
    x: np.ndarray
    masken: np.ndarray               # alle Zeilen (Einer-Koalitionen zuerst)
    werte: np.ndarray
    schleifen: int
    n_positiv: int
    abgebrochen: bool
    t_lp: float
    t_trennung: float
    verlauf: list

    @property
    def zeilen(self) -> int:
        return len(self.masken)


def row_generation(game, separator, k: int = 1, max_schleifen: int | None = None, verlauf: bool = True) -> RowgenErgebnis:
    """`separator(game, x, k) -> [(Maske, Überschuss), ...]` (absteigend); k Zeilen je Schleife höchstens."""
    n = game.n
    if n < 2:   # keine echte Koalition: ε* = 0 per Definition, x = (v(N))
        return RowgenErgebnis(0.0, np.array([game.vN]), np.zeros((0, n), dtype=bool), np.zeros(0), 0, 0, False, 0.0, 0.0, [])
    S0, v0 = LP.einer_zeilen(game)
    masken, werte = list(S0), list(v0)
    gesehen = {G.schluessel(s) for s in masken}
    tol = REL_TOL * max(1.0, abs(game.vN))
    t_lp = t_trennung = 0.0
    protokoll: list[SchleifenSchritt] = []
    schleife = 0
    while True:
        schleife += 1
        t0 = time.perf_counter()
        lp = LP.least_core_lp(game, masken, werte)
        t_lp += time.perf_counter() - t0
        t0 = time.perf_counter()
        treffer = separator(game, lp.x, k)
        t_trennung += time.perf_counter() - t0
        neu = [(S, e) for S, e in treffer if e > lp.eps + tol and G.schluessel(S) not in gesehen]
        schritt = SchleifenSchritt(len(masken), lp.eps)
        if not neu:
            if verlauf:
                protokoll.append(schritt)
            return RowgenErgebnis(lp.eps, lp.x, np.array(masken), np.array(werte), schleife, lp.n_positiv, False, t_lp, t_trennung, protokoll)
        if max_schleifen is not None and schleife >= max_schleifen:
            if verlauf:
                protokoll.append(schritt)
            return RowgenErgebnis(lp.eps, lp.x, np.array(masken), np.array(werte), schleife, lp.n_positiv, True, t_lp, t_trennung, protokoll)
        S_neu = np.array([S for S, _ in neu])
        v_neu = game.value_batch(S_neu, count=False)
        for S, v in zip(S_neu, v_neu):
            masken.append(S)
            werte.append(float(v))
            gesehen.add(G.schluessel(S))
        if verlauf:
            schritt.schnitte = [(S.copy(), float(e)) for S, e in neu]
            protokoll.append(schritt)


def standard_separator(game):
    """Die exakte Trennung der Spielklasse: Rucksack-MILP (Gewicht) bzw. Aufzählung der Standortmengen."""
    import kzl_separation as SEP
    return SEP.separate_knapsack if game.kind == "gewinn" else SEP.separate_facility_enum
