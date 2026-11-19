"""Verfahren, die Koalitionen ZIEHEN statt gezielt zu suchen, und ihr Maßstab (Einheit je Verfahren).

random_coalitions   K zufällige echte nichtleere Koalitionen ("halb": jedes Mitglied mit Wahrscheinlichkeit 1/2; "gleich": Größe gleichverteilt in 1..n-1).
sample_lp           Stichproben-LP: nur die Einer-Koalitionen und K gezogene Koalitionen als Zeilen, ein LP.
core_lagrangian     Eigener Nachbau eines Verfahrens nach der Beschreibung bei Gemp u. a. (2024), aus einem Textauszug: Mirror Prox auf ε + λ (Verlust - δ²/2) mit Stapeln
                    zufälliger Koalitionen. Hyperparameter geraten bzw. abgestimmt; die Algorithmen und der Code der Autoren sind NICHT nachgebaut oder gelesen.
targeted_lp         Gezielte Heuristik: LP, dazu eine heuristische Trennung (Rucksack-Greedy bzw. Einzelflip-Suche); keine Garantie.
true_violation      Größter Überschuss einer Aufteilung über ALLE echten nichtleeren Koalitionen (exakte Trennung).
estimated_violation Dieselbe Größe geschätzt als größter Überschuss über 50 000 Zufallskoalitionen (so wird in Sampling-Arbeiten der Fehler geschätzt).
Maß für die Güte: Lücke = (wahre Verletzung - ε*) / Wert(N); Lücke 0 heißt, die gelieferte Aufteilung liegt im Least Core."""

import numpy as np

import kzl_constants as C
import kzl_games as G
import kzl_lp as LP
import kzl_rowgen as RG
import kzl_separation as SEP


def random_coalitions(stream, n: int, K: int, groesse: str = "halb") -> np.ndarray:
    """K zufällige echte nichtleere Koalitionen (K x n, Bool); n >= 2."""
    if n < 2:
        raise ValueError("random_coalitions: n >= 2")
    ergebnis = np.zeros((0, n), dtype=bool)
    while len(ergebnis) < K:
        fehlt = K - len(ergebnis)
        if groesse == "halb":
            S = stream.matrix(fehlt, n) < 0.5
        else:
            u = stream.block(fehlt)
            groessen = 1 + np.minimum(n - 2, np.floor(u * (n - 1)).astype(int))   # 1..n-1
            rang = np.argsort(np.argsort(stream.matrix(fehlt, n), axis=1), axis=1)
            S = rang < groessen[:, None]
        sz = S.sum(axis=1)
        ergebnis = np.vstack([ergebnis, S[(sz > 0) & (sz < n)]])
    return ergebnis[:K]


def sample_lp(game, K: int, stream, groesse: str = "halb"):
    """Rückgabe: (ε des Stichproben-LP, x, Zahl der Zeilen)."""
    S0, v0 = LP.einer_zeilen(game)
    S = random_coalitions(stream, game.n, K, groesse)
    v = game.value_batch(S)
    lp = LP.least_core_lp(game, np.vstack([S0, S]), np.concatenate([v0, v]))
    return lp.eps, lp.x, len(S0) + len(S)


def core_lagrangian(game, iterationen: int, stapel: int, stream, eta0: float = C.DEFAULT_ETA, delta: float = C.CL_DELTA):
    """Mirror Prox (Extragradient) auf min_{x im Simplex, ε} max_{λ in [0, n/δ]}  ε + λ (E[max(0, e/N - ε)² / (2|S|)] - δ²/2); N = Wert der großen Koalition.
    Rückgabe: (x, ε) als gewichtetes Mittel der Iterierten, auf Wert(N) zurückskaliert."""
    n = game.n
    N = game.vN
    vorz = 1.0 if game.kind == "kosten" else -1.0
    lam_max = n / delta
    x = np.full(n, 1.0 / n)
    eps = 0.0
    lam = 1.0

    def karte(x, eps, lam):
        S = random_coalitions(stream, n, stapel, "halb")
        sz = S.sum(axis=1).astype(float)
        e = game.excess(x * N, S) / N
        r = np.maximum(0.0, e - eps)
        w = r / sz
        gx = vorz * lam * (S.astype(float) * w[:, None]).mean(axis=0)
        ge = 1.0 - lam * w.mean()
        gl = -((r * r / (2 * sz)).mean() - delta ** 2 / 2)
        return gx, ge, gl

    def schritt(x, eps, lam, g, eta):
        gx, ge, gl = g
        x2 = x * np.exp(-eta * gx)
        x2 = x2 / x2.sum()
        return x2, float(np.clip(eps - eta * ge, -1.0, 1.0)), float(np.clip(lam - eta * gl, 0.0, lam_max))

    xs = np.zeros(n)
    es = 0.0
    wsum = 0.0
    for t in range(1, iterationen + 1):
        eta = eta0 / np.sqrt(t)
        g1 = karte(x, eps, lam)
        xh, eh, lh = schritt(x, eps, lam, g1, eta)
        g2 = karte(xh, eh, lh)
        x, eps, lam = schritt(x, eps, lam, g2, eta)
        xs += eta * x
        es += eta * eps
        wsum += eta
    return xs / wsum * N, es / wsum * N


def _heuristik_standort(game, x, stream, starts: int, beste: int):
    """Zufallsstarts mit Einzelflip-Suche auf e(S, x); die besten verschiedenen Koalitionen (Maske, Überschuss)."""
    n = game.n
    u = stream.block(starts * (n + 1))
    p = 0.2 + 0.6 * u[:starts]
    zufall = u[starts:].reshape(starts, n)
    gefunden = {}
    for s in range(starts):
        S = zufall[s] < p[s]
        if S.sum() == 0:
            S[int(np.argmin(zufall[s]))] = True
        if S.all():
            S[int(np.argmax(zufall[s]))] = False
        e = float(game.excess(x, S[None, :])[0])
        for _ in range(3 * n):
            nb = np.tile(S, (n, 1))
            nb[np.arange(n), np.arange(n)] ^= True
            ok = (nb.sum(axis=1) > 0) & (nb.sum(axis=1) < n)
            ev = np.full(n, -np.inf)
            ev[ok] = game.excess(x, nb[ok])
            j = int(np.argmax(ev))
            if ev[j] > e + 1e-12:
                S, e = nb[j], float(ev[j])
            else:
                break
        gefunden[G.schluessel(S)] = (S, e)
    return sorted(gefunden.values(), key=lambda t: -t[1])[:beste]


def _heuristik_gewicht(game, x, stream, starts: int, beste: int):
    """Rucksack-Greedy mit zufälliger Störung: nach x_i / w_i aufsteigend auffüllen, bis w(S) >= q; dann Überflüssige mit großem x_i streichen."""
    n = game.n
    rauschen = 0.7 + 0.6 * stream.matrix(starts, n)
    gefunden = {}
    for s in range(starts):
        verhaeltnis = (x + 1e-12) / game.w * rauschen[s]
        S = np.zeros(n, dtype=bool)
        summe = 0.0
        for i in np.argsort(verhaeltnis, kind="stable"):
            S[i] = True
            summe += game.w[i]
            if summe >= game.q - 0.5:
                break
        for i in np.argsort(-x, kind="stable"):
            if S[i] and summe - game.w[i] >= game.q - 0.5:
                S[i] = False
                summe -= game.w[i]
        if S.all() or S.sum() == 0:
            continue
        gefunden[G.schluessel(S)] = (S, float(game.excess(x, S[None, :])[0]))
    return sorted(gefunden.values(), key=lambda t: -t[1])[:beste]


def targeted_lp(game, stream, starts: int = C.GEZIELT_STARTS, beste: int = C.GEZIELT_BEST, max_schleifen: int = C.MAX_SCHLEIFEN_LIVE):
    """Rückgabe: (ε, x, Zeilen, Schleifen). Hört auf, sobald die Heuristik keine verletzte Koalition mehr findet (das beweist nichts)."""
    if game.n < 2:
        return 0.0, np.array([game.vN]), 0, 0
    heuristik = _heuristik_standort if game.kind == "kosten" else _heuristik_gewicht
    S0, v0 = LP.einer_zeilen(game)
    masken, werte = list(S0), list(v0)
    gesehen = {G.schluessel(s) for s in masken}
    tol = RG.REL_TOL * max(1.0, abs(game.vN))
    for schleife in range(1, max_schleifen + 1):
        lp = LP.least_core_lp(game, masken, werte)
        neu = [(S, e) for S, e in heuristik(game, lp.x, stream, starts, beste) if e > lp.eps + tol and G.schluessel(S) not in gesehen]
        if not neu:
            return lp.eps, lp.x, len(masken), schleife
        for S, _ in neu:
            masken.append(S)
            werte.append(float(game.value_batch(S[None, :])[0]))
            gesehen.add(G.schluessel(S))
    return lp.eps, lp.x, len(masken), max_schleifen


def true_violation(game, x) -> float:
    """Größter Überschuss von x über alle echten nichtleeren Koalitionen (exakt); beim Gewinnspiel zusätzlich die Imputation x_i >= 0 (negative Anteile)."""
    if game.n < 2:
        return 0.0
    trenner = SEP.separate_knapsack if game.kind == "gewinn" else SEP.separate_facility_enum
    treffer = trenner(game, x, 1)
    if game.kind == "gewinn":
        e = treffer[0][1] if treffer else -np.inf    # ohne beschlussfähige echte Koalition bleibt nur die Imputationsgrenze
        return float(max(e, float(np.max(-np.asarray(x)))))
    return float(treffer[0][1])


def estimated_violation(game, x, stream, anzahl: int = C.FEHLER_KOALITIONEN) -> float:
    """Größter Überschuss über `anzahl` Zufallskoalitionen (jedes Mitglied mit Wahrscheinlichkeit 1/2); stets höchstens die wahre Verletzung."""
    S = random_coalitions(stream, game.n, anzahl, "halb")
    return float(np.max(game.excess(x, S)))
