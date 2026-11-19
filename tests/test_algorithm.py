"""Kern, Einheit für Einheit an den Mini-Instanzen aus conftest.py (von Hand gerechnet, DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import kzl_games as G
import kzl_lp as LP
import kzl_power as PW
import kzl_rowgen as RG
import kzl_sampling as SA
import kzl_scenario as SC
import kzl_separation as SEP
from kzl_stream import Stream


def _maske(n, *mitglieder):
    m = np.zeros(n, dtype=bool)
    m[list(mitglieder)] = True
    return m


# --- Spiele -------------------------------------------------------------------------------------------------------------------------------

def test_gewichtsspiel_gewinner_von_hand(hand4):
    gewinner = {tuple(np.flatnonzero(S)) for S in G.alle_koalitionen(4) if hand4.value_batch(S[None, :], count=False)[0] == 1.0}
    assert gewinner == {(0, 1), (0, 2), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}


def test_gewichtsspiel_ueberschuss_und_veto(hand4, drei_gleiche):
    x = np.array([0.4, 0.3, 0.2, 0.1])
    assert hand4.excess(x, _maske(4, 0, 2)[None, :])[0] == pytest.approx(1 - 0.6)
    assert not hand4.has_veto_player()                                   # {2,3,4} gewinnt ohne Spieler 1
    assert not drei_gleiche.has_veto_player()
    assert G.GewichtsSpiel([6, 1, 1], 5).has_veto_player()               # ohne Spieler 1 sind es nur 2 < 5
    assert G.GewichtsSpiel([6, 1, 1], 5).vN == 1.0 and G.GewichtsSpiel([1, 1], 5).vN == 0.0


def test_standortspiel_kosten_von_hand(zwei_kunden, dreieck3):
    assert zwei_kunden.value_batch(np.array([[True, False], [False, True], [True, True]]), count=False).tolist() == [12.0, 14.0, 16.0]
    assert zwei_kunden.vN == 16.0 and zwei_kunden.excess(np.array([7.0, 9.0]), np.array([[True, False]]))[0] == pytest.approx(7 - 12)
    assert dreieck3.vN == 60.0
    assert dreieck3.value_batch(np.array([[True, True, False], [True, False, False]]), count=False).tolist() == [30.0, 30.0]
    assert len(dreieck3.open_facilities()) == 2


def test_zaehler_der_koalitionswerte(hand4):
    hand4.n_eval = 0
    hand4.value_batch(np.ones((5, 4), dtype=bool))
    hand4.value_batch(np.ones((7, 4), dtype=bool), count=False)
    assert hand4.n_eval == 5


# --- LP ----------------------------------------------------------------------------------------------------------------------------------

def test_lp_mit_den_drei_wirksamen_zeilen_von_hand(hand4):
    rows = np.array([_maske(4, 0, 1), _maske(4, 0, 2), _maske(4, 1, 2, 3)] + list(np.eye(4, dtype=bool)))
    lp = LP.least_core_lp(hand4, rows, hand4.value_batch(rows, count=False))
    assert lp.eps == pytest.approx(1 / 3, abs=1e-9)
    assert lp.x[3] == pytest.approx(0.0, abs=1e-9) and lp.x.sum() == pytest.approx(1.0)


def test_lp_kostenspiel_mit_innerem(zwei_kunden):
    lp = LP.full_lp(zwei_kunden)
    assert lp.eps == pytest.approx(-5.0, abs=1e-9)
    assert lp.x == pytest.approx([7.0, 9.0], abs=1e-9)


def test_lp_ohne_nennenswerte_zeilen_ist_untere_schranke(hand4):
    """Mit weniger Zeilen ist ε höchstens ε*: nur die Einer-Koalitionen geben -1/4 (x = 1/4 je Spieler)."""
    S, v = LP.einer_zeilen(hand4)
    assert LP.least_core_lp(hand4, S, v).eps == pytest.approx(-0.25, abs=1e-9)


# --- Trennung -----------------------------------------------------------------------------------------------------------------------------

def test_rucksack_trennung_von_hand(hand4):
    S, e = SEP.separate_knapsack(hand4, [0.4, 0.3, 0.2, 0.1])[0]
    assert e == pytest.approx(0.4)                                       # kleinste Summe x(S) unter w(S) >= 6 ist 0,6: {1,3} oder {2,3,4}
    assert tuple(np.flatnonzero(S)) in {(0, 2), (1, 2, 3)}


def test_standort_trennung_aufzaehlung_und_milp_von_hand(dreieck3):
    x = np.array([20.0, 20.0, 20.0])
    for trenner in (SEP.separate_facility_enum, SEP.separate_facility_milp):
        S, e = trenner(dreieck3, x)[0]
        assert e == pytest.approx(40.0 - 30.0) and S.sum() == 2          # Paar: x(S) = 40, c(S) = 30


def test_trennung_liefert_hoechstens_k_verschiedene_koalitionen(dreieck3):
    treffer = SEP.separate_facility_enum(dreieck3, np.array([20.0, 20.0, 20.0]), k=3)
    assert 1 <= len(treffer) <= 3
    assert len({G.schluessel(S) for S, _ in treffer}) == len(treffer)
    assert all(treffer[i][1] >= treffer[i + 1][1] for i in range(len(treffer) - 1))


# --- Zeilenerzeugung -----------------------------------------------------------------------------------------------------------------------

def test_zeilenerzeugung_handbeispiele(hand4, drei_gleiche, dreieck3, zwei_kunden):
    r = RG.row_generation(hand4, SEP.separate_knapsack)
    assert r.eps == pytest.approx(1 / 3, abs=1e-8) and not r.abgebrochen and r.zeilen <= 4 + 3
    assert RG.row_generation(drei_gleiche, SEP.separate_knapsack).eps == pytest.approx(1 / 3, abs=1e-8)
    assert RG.row_generation(dreieck3, SEP.separate_facility_enum).eps == pytest.approx(10.0, abs=1e-6)
    assert RG.row_generation(zwei_kunden, SEP.separate_facility_enum).eps == pytest.approx(-5.0, abs=1e-6)


def test_zeilenerzeugung_protokoll_ist_schluessig(hand4):
    r = RG.row_generation(hand4, SEP.separate_knapsack)
    assert r.verlauf[0].zeilen_vorher == 4 and r.zeilen == 4 + sum(len(s.schnitte) for s in r.verlauf)
    assert all(a.eps <= b.eps + 1e-9 for a, b in zip(r.verlauf, r.verlauf[1:]))   # mehr Zeilen, größeres ε
    assert r.verlauf[-1].schnitte == []                                           # in der letzten Schleife wird nichts mehr verletzt
    assert r.schleifen == len(r.verlauf)


def test_obergrenze_wird_ausgewiesen_und_gibt_untere_schranke(dreieck3):
    exakt = RG.row_generation(dreieck3, SEP.separate_facility_enum, 1)
    kurz = RG.row_generation(dreieck3, SEP.separate_facility_enum, 1, max_schleifen=1)
    assert kurz.abgebrochen and kurz.eps <= exakt.eps + 1e-9 and kurz.schleifen == 1
    assert not exakt.abgebrochen


# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------

def test_quote_ist_exakt_in_ganzen_zahlen():
    assert SC.quote_of(np.full(30, 10), 50) == 151                       # mehr als 150
    assert SC.quote_of(np.full(30, 1), 67) == 21                         # 30 * 67 / 100 = 20,1 -> mehr als 20,1, also 21
    assert SC.quote_of([4, 3, 2, 1], 50) == 6                            # mehr als 5
    assert SC.quote_of([10, 10, 10], 60) == 19                           # 18 ist nicht "mehr als 18": 19


def test_gewichts_erzeuger_bereiche_und_festheit():
    for er, lo, hi in (("gestuft", 5, 40), ("gleichverteilt", 1, 100)):
        w = SC.weights(er, 400, 3)
        assert w.min() >= lo and w.max() <= hi
    assert set(SC.weights("gestuft", 400, 3)) == {5, 10, 20, 40}
    assert SC.weights("gleich", 7, 3).tolist() == [10] * 7
    p = SC.weights("pareto", 400, 3)
    assert p.min() >= 10 and p.max() > 40                       # Tail: Pareto(1,2)*10 >= 10, einzelne Werte weit darüber
    assert SC.weights("pareto", 50, 3).tolist() == SC.weights("pareto", 50, 3).tolist()
    assert SC.weights("pareto", 50, 3).tolist() != SC.weights("pareto", 50, 4).tolist()


def test_eigene_liste_wird_geprueft():
    assert SC.parse_liste("4, 3,2;1") == [4, 3, 2, 1]
    assert SC.parse_liste("4") is None and SC.parse_liste("a,b") is None and SC.parse_liste("0,3") is None
    assert SC.parse_liste(",".join(["5"] * 101)) is None and SC.parse_liste("1001,5") is None
    assert SC.weights("liste", 99, 0, "kaputt").tolist() == [4, 3, 2, 1]  # ungültig: Rückfall auf das Handbeispiel


def test_dreiecks_typen_rein_und_zufall():
    g = SC.triangle_game(30, 30.0)
    assert g.m == 3 and (g.d == 1000.0).sum() == 30 and (g.d[np.arange(30), np.arange(30) % 3] == 1000.0).all()
    z = SC.triangle_random_game(5, 30, 30.0)
    assert (z.d == 1000.0).sum() == 30 and (z.d[z.d != 1000.0] < 2.0).all() and (0.9 * 30 <= z.fix).all() and (z.fix <= 1.1 * 30).all()


def test_stromblock_ist_reproduzierbar_und_teilstroeme_sind_verschieden():
    a, b = Stream(7), Stream(7)
    assert np.array_equal(a.block(10), b.block(10)) and np.array_equal(a.block(5), b.block(5))
    s = Stream(7)
    assert not np.array_equal(s.block(10), s.block(10))


# --- Stichproben ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("groesse", ["halb", "gleich"])
def test_zufallskoalitionen_sind_echt_und_nichtleer(groesse):
    S = SA.random_coalitions(Stream(1), 6, 300, groesse)
    assert S.shape == (300, 6) and (S.sum(axis=1) >= 1).all() and (S.sum(axis=1) <= 5).all()
    assert set(S.sum(axis=1)) == {1, 2, 3, 4, 5} if groesse == "gleich" else set(S.sum(axis=1)) >= {2, 3, 4}


def test_stichproben_lp_ist_untere_schranke_und_wahre_verletzung_mindestens_eps_stern(hand4):
    eps, x, zeilen = SA.sample_lp(hand4, 3, Stream(2))
    assert eps <= 1 / 3 + 1e-9 and zeilen == 4 + 3
    assert SA.true_violation(hand4, x) >= 1 / 3 - 1e-9                    # keine Aufteilung hat eine kleinere größte Verletzung als ε*


def test_wahre_verletzung_der_optimalen_aufteilung_ist_eps_stern(hand4, dreieck3):
    for game, soll in ((hand4, 1 / 3), (dreieck3, 10.0)):
        r = RG.row_generation(game, RG.standard_separator(game), 1)
        assert SA.true_violation(game, r.x) == pytest.approx(soll, abs=1e-6)


def test_geschaetzte_verletzung_ist_hoechstens_die_wahre(dreieck3):
    x = np.array([20.0, 20.0, 20.0])
    assert SA.estimated_violation(dreieck3, x, Stream(3), 500) <= SA.true_violation(dreieck3, x) + 1e-12


def test_gezielte_heuristik_findet_das_handbeispiel(hand4, dreieck3):
    for game, soll in ((hand4, 1 / 3), (dreieck3, 10.0)):
        eps, x, zeilen, schleifen = SA.targeted_lp(game, Stream(4))
        assert eps == pytest.approx(soll, abs=1e-6) and zeilen >= game.n and schleifen >= 1


def test_core_lagrangian_liefert_aufteilung_mit_wert_der_grossen_koalition(hand4):
    x, eps = SA.core_lagrangian(hand4, 200, 32, Stream(5), eta0=3.0)
    assert x.sum() == pytest.approx(1.0) and (x >= 0).all()


# --- Shapley–Shubik ------------------------------------------------------------------------------------------------------------------------

def test_shapley_shubik_handbeispiel_in_zwoelfteln(hand4):
    assert PW.shapley_shubik(hand4.w, hand4.q) * 12 == pytest.approx([5, 3, 3, 1], abs=1e-12)


def test_shapley_shubik_summiert_zu_eins_und_ist_symmetrisch():
    phi = PW.shapley_shubik(np.full(6, 10), 31)
    assert phi.sum() == pytest.approx(1.0) and phi == pytest.approx([1 / 6] * 6)
