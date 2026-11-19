"""Orakel-Tests (anderer Rechenweg, DEMO-PLAYBOOK §12): die Zeilenerzeugung gegen das volle LP über alle 2^n - 2 Koalitionen, Rucksack-MILP gegen Rucksack-DP,
Standort-Trennung per Aufzählung gegen MILP, geschlossene Formeln (gleiche Gewichte, Dreiecks-Typen), Veto-Regel, Ganzzahligkeitslücke, Shapley-Shubik per Aufzählung.

Die Vergleichsfunktionen sind Module-Funktionen mit der Instanzenfolge als Argument: der Scratch-Lauf (mehr als 200 Instanzen je Vergleich) ruft sie mit längeren
Folgen auf, das Repo prüft die kurze Fassung. Zählungen werden mit ausgewiesen, damit kein Vergleich leer bleibt (kein toter Zweig)."""

import math

import numpy as np
import pytest

import kzl_games as G
import kzl_lp as LP
import kzl_power as PW
import kzl_rowgen as RG
import kzl_scenario as SC
import kzl_separation as SEP
from kzl_stream import Stream

TOL = 1e-6
ERZEUGER = ("gleich", "gestuft", "pareto", "gleichverteilt")


# --- Vergleichsfunktionen (auch vom Scratch-Lauf benutzt) -------------------------------------------------------------------------------------------------

def verletzung_ueber_alle(game, x) -> float:
    """Größter Überschuss von x über ALLE 2^n - 2 Koalitionen (Aufzählung), bei Gewinnspielen auch über die Imputationsgrenze x >= 0."""
    S = G.alle_koalitionen(game.n)
    e = float(game.excess(x, S).max())
    return max(e, float(np.max(-x))) if game.kind == "gewinn" else e


def vergleiche_gewicht(seeds, n: int, prozente=(50, 60, 67)):
    """Zeilenerzeugung (Rucksack-MILP) gegen volles LP und gegen die Veto-Handregel. Rückgabe: (Vergleiche, Abweichungen, Instanzen mit leerem Kern)."""
    vergleiche = abweichungen = leer = 0
    for er in ERZEUGER:
        for pct in prozente:
            for sd in seeds:
                g = SC.weighted_game(er, n, sd, pct)
                r = RG.row_generation(g, SEP.separate_knapsack)
                voll = LP.full_lp(g).eps
                ok = abs(r.eps - voll) < TOL                                              # gleicher Least-Core-Wert
                ok = ok and abs(verletzung_ueber_alle(g, r.x) - voll) < TOL               # die gelieferte Aufteilung hält ALLE Bedingungen
                ok = ok and ((voll > TOL) == (not g.has_veto_player()))                  # Kern leer <=> kein Veto-Spieler
                vergleiche += 1
                abweichungen += not ok
                leer += voll > TOL
    return vergleiche, abweichungen, leer


def vergleiche_standort(seeds, n: int, m: int, fix: float, deckel: float | None):
    """Zeilenerzeugung mit Aufzählungs- und mit MILP-Trennung gegen das volle LP (Kostenspiel). Rückgabe: (Vergleiche, Abweichungen, Instanzen mit leerem Kern)."""
    vergleiche = abweichungen = leer = 0
    for sd in seeds:
        g = SC.euclid_game(sd, n, m, fix, deckel)
        voll = LP.full_lp(g).eps
        a = RG.row_generation(g, SEP.separate_facility_enum, 4)
        b = RG.row_generation(g, SEP.separate_facility_milp, 1)
        tol = TOL * max(1.0, g.vN)
        ok = abs(a.eps - voll) < tol and abs(b.eps - voll) < tol and abs(verletzung_ueber_alle(g, a.x) - voll) < tol
        vergleiche += 2
        abweichungen += (not ok) * 2
        leer += voll > tol
    return vergleiche, abweichungen, leer


def rucksack_dp(x, w, q) -> float:
    """Kleinstes x(S) unter w(S) >= q per DP über die Gewichtssumme (anderer Rechenweg als das MILP); S = N zugelassen."""
    W = int(round(q))
    best = [math.inf] * (W + 1)
    best[0] = 0.0
    for xi, wi in zip(x, [int(v) for v in w]):
        neu = best[:]
        for t in range(W + 1):
            if best[t] < math.inf:
                t2 = min(W, t + wi)
                neu[t2] = min(neu[t2], best[t] + xi)
        best = neu
    return best[W]


def ufl_relaxation(game) -> float:
    """Wert der LP-Relaxation des Standortproblems (min Fixkosten + Wege; z_ij <= y_j); anderer Rechenweg als die Koalitionsfunktion."""
    from scipy.optimize import linprog
    n, m = game.n, game.m
    c = np.concatenate([game.fix, game.d.ravel()])
    A_ub, b_ub = [], []
    for i in range(n):
        for j in range(m):
            row = np.zeros(m + n * m)
            row[m + i * m + j] = 1
            row[j] = -1
            A_ub.append(row)
            b_ub.append(0.0)
    A_eq = np.zeros((n, m + n * m))
    for i in range(n):
        A_eq[i, m + i * m:m + (i + 1) * m] = 1
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
    return float(res.fun)


def shapley_shubik_aufzaehlung(w, q):
    """Orakel: Shapley-Shubik über alle Teilmengen (n <= 12): Spieler i ist Dreh- und Angelpunkt, wenn w(S) < q <= w(S) + w_i."""
    n = len(w)
    phi = np.zeros(n)
    for mask in range(1 << n):
        S = [j for j in range(n) if (mask >> j) & 1]
        ws = sum(w[j] for j in S)
        for i in range(n):
            if not (mask >> i) & 1 and ws < q <= ws + w[i]:
                phi[i] += math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
    return phi


# --- Tests ----------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 9])
def test_gewichtsspiel_zeilenerzeugung_gleich_volle_lp_und_veto_regel(n):
    vergleiche, abweichungen, leer = vergleiche_gewicht(range(2), n)
    assert vergleiche == 4 * 3 * 2 and abweichungen == 0
    assert leer >= 1                                                       # der Vergleich ist nicht leer: es kommen leere Kerne vor


def test_gewichtsspiel_mit_nicht_leerem_kern_kommt_vor():
    """Die Veto-Regel wird nur im Zweig 'Kern nicht leer' geprüft, wenn dieser Fall wirklich vorkommt (hier: Handbeispiel mit Veto-Spieler)."""
    g = G.GewichtsSpiel([50, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 51)
    assert g.has_veto_player() and LP.full_lp(g).eps == pytest.approx(0.0, abs=TOL)
    assert RG.row_generation(g, SEP.separate_knapsack).eps == pytest.approx(0.0, abs=TOL)


def test_standortspiel_zeilenerzeugung_gleich_volle_lp_mit_beiden_trennungen():
    vergleiche, abweichungen, leer = vergleiche_standort(range(8), 8, 5, 60.0, None)
    assert vergleiche == 16 and abweichungen == 0
    v2, a2, _ = vergleiche_standort(range(8), 8, 5, 60.0, 30.0)             # mit Zukaufpreis-Deckel
    assert a2 == 0 and v2 == 16
    assert leer == 0                                                       # in dieser Familie ist der Kern bei n = 8 nie leer: der Zweig 'leer' kommt unten (Dreiecks-Typen)


def test_standortspiel_mit_leerem_kern_zeilenerzeugung_gleich_volle_lp():
    leer = 0
    for sd in range(5):
        g = SC.triangle_random_game(sd, 9, 30.0)
        voll = LP.full_lp(g).eps
        for r in (RG.row_generation(g, SEP.separate_facility_enum, 4), RG.row_generation(g, SEP.separate_facility_milp)):
            assert r.eps == pytest.approx(voll, abs=1e-6)
            assert verletzung_ueber_alle(g, r.x) == pytest.approx(voll, abs=1e-6)
        leer += voll > 1e-6
    assert leer == 5


def test_rucksack_milp_gleich_rucksack_dp():
    for sd in range(12):
        s = Stream(sd)
        g = SC.weighted_game(ERZEUGER[sd % 4], 12, sd, 50)
        x = s.block(12)
        x = x / x.sum()
        _, e = SEP.separate_knapsack(g, x)[0]
        assert 1.0 - rucksack_dp(x, g.w, g.q) == pytest.approx(e, abs=1e-9), sd


def test_standort_trennung_aufzaehlung_gleich_milp_im_wert():
    gleich = 0
    for sd in range(10):
        g = SC.euclid_game(sd, 10, 6, 60.0)
        x = np.array(Stream(sd).block(10)) * g.vN / 5.0
        e_enum = SEP.separate_facility_enum(g, x, 4)[0][1]
        e_milp = SEP.separate_facility_milp(g, x)[0][1]
        gleich += abs(e_enum - e_milp) < 1e-6
    assert gleich == 10


def test_gleiche_gewichte_geschlossene_formel():
    for n in (3, 5, 8, 11, 14, 20, 30, 31):
        g = G.GewichtsSpiel(np.ones(n), n // 2 + 1)
        soll = 1 - (n // 2 + 1) / n
        assert RG.row_generation(g, SEP.separate_knapsack).eps == pytest.approx(soll, abs=1e-7), n
    assert 1 - (30 // 2 + 1) / 30 == pytest.approx(14 / 30)


def test_dreiecks_typen_geschlossene_formel_und_volle_lp():
    for n in (3, 6, 9):
        g = SC.triangle_game(n, 30.0)
        assert LP.full_lp(g).eps == pytest.approx(10.0, abs=1e-6), n
        assert RG.row_generation(g, SEP.separate_facility_enum, 3).eps == pytest.approx(10.0, abs=1e-6)
        assert RG.row_generation(g, SEP.separate_facility_milp).eps == pytest.approx(10.0, abs=1e-6)
    assert RG.row_generation(SC.triangle_game(30, 30.0), SEP.separate_facility_enum, 5).eps == pytest.approx(10.0, abs=1e-5)


def test_kern_leer_genau_dann_wenn_ganzzahligkeitsluecke():
    """Standortspiel: Kern leer <=> die LP-Relaxation liegt echt unter c(N). Gezählt beide Fälle: Dreiecks-Typen (Lücke) und euklidische Lagen (meist keine)."""
    mit_luecke = ohne_luecke = 0
    spiele = [SC.triangle_game(n, 30.0) for n in (3, 4, 6)] + [SC.triangle_random_game(sd, 7, 30.0) for sd in range(3)]
    spiele += [SC.euclid_game(sd, 9, 5, 60.0, 30.0) for sd in range(8)]
    for g in spiele:
        luecke = g.vN - ufl_relaxation(g) > 1e-6 * g.vN
        leer = RG.row_generation(g, SEP.separate_facility_enum, 4).eps > 1e-6 * g.vN
        assert luecke == leer
        mit_luecke += luecke
        ohne_luecke += not luecke
    assert mit_luecke >= 3 and ohne_luecke >= 1


def test_shapley_shubik_dp_gleich_aufzaehlung():
    for sd in range(6):
        w = SC.weights(ERZEUGER[sd % 4], 10, sd)
        q = SC.quote_of(w, 50)
        assert PW.shapley_shubik(w, q) == pytest.approx(shapley_shubik_aufzaehlung(list(w), q), abs=1e-12)


def test_eigenschaften_umnummerieren_und_skalieren():
    """ε* ändert sich nicht, wenn man die Spieler umnummeriert; bei Kosten mal c > 0 skaliert es mit c."""
    g = SC.euclid_game(3, 12, 5, 60.0)
    perm = np.array([5, 11, 0, 3, 8, 1, 10, 2, 7, 4, 9, 6])
    g2 = G.StandortSpiel(g.d[perm], g.fix)
    g3 = G.StandortSpiel(g.d * 2.5, g.fix * 2.5)
    e = RG.row_generation(g, SEP.separate_facility_enum, 4).eps
    assert RG.row_generation(g2, SEP.separate_facility_enum, 4).eps == pytest.approx(e, abs=1e-6)
    assert RG.row_generation(g3, SEP.separate_facility_enum, 4).eps == pytest.approx(2.5 * e, abs=1e-5)
    w = SC.weights("gestuft", 14, 2)
    a = RG.row_generation(G.GewichtsSpiel(w, SC.quote_of(w, 60)), SEP.separate_knapsack).eps
    b = RG.row_generation(G.GewichtsSpiel(w[::-1], SC.quote_of(w, 60)), SEP.separate_knapsack).eps
    assert a == pytest.approx(b, abs=1e-8)
