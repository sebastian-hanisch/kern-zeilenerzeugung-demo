"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, inklusive Grenzen, kein Gewinner außer N."""

import numpy as np
import pytest

import kzl_evaluation as EV
import kzl_games as G
import kzl_lp as LP
import kzl_power as PW
import kzl_rowgen as RG
import kzl_sampling as SA
import kzl_scenario as SC
import kzl_separation as SEP
from kzl_stream import Stream


def _ze(game, k=1):
    return RG.row_generation(game, RG.standard_separator(game), k)


def test_ein_spieler_hat_keine_echte_koalition():
    for game in (G.GewichtsSpiel([5], 3), G.StandortSpiel(np.array([[1.0, 2.0]]), np.array([4.0, 4.0]))):
        r = _ze(game)
        assert r.eps == 0.0 and r.zeilen == 0 and r.x[0] == pytest.approx(game.vN) and not r.abgebrochen
        assert LP.full_lp(game).eps == 0.0
        assert EV.alle_bedingungen(1) == 0
    assert EV.verdict_kern("gewinn", "rand", 0.0, 1.0, 1)[0] == "info"


def test_zwei_spieler_gewichtsspiel_beide_mit_veto_hat_kern_mit_innerem():
    g = G.GewichtsSpiel([1, 1], SC.quote_of([1, 1], 50))               # Quote 2: nur beide zusammen gewinnen
    r = _ze(g)
    assert g.has_veto_player() and r.eps == pytest.approx(-0.5, abs=1e-9)
    assert EV.core_class(r.eps, g.vN) == "inneres"
    assert r.x == pytest.approx([0.5, 0.5])


def test_zwei_spieler_diktator_kern_am_rand():
    g = G.GewichtsSpiel([3, 1], SC.quote_of([3, 1], 50))               # Quote 3: Spieler 1 gewinnt allein
    r = _ze(g)
    assert r.eps == pytest.approx(0.0, abs=1e-9) and r.x == pytest.approx([1.0, 0.0], abs=1e-9)
    assert EV.core_class(r.eps, g.vN) == "rand" and g.has_veto_player()


def test_zwei_kunden_standortspiel():
    g = G.StandortSpiel(np.array([[2.0, 9.0], [9.0, 2.0]]), np.array([10.0, 10.0]))
    r = _ze(g)
    assert r.eps == pytest.approx(LP.full_lp(g).eps, abs=1e-9) and r.zeilen == 2


def test_kein_gewinner_ausser_der_grossen_koalition():
    """Quote gleich der Gewichtssumme: die Rucksack-Trennung ist unzulässig (keine echte Koalition gewinnt), die Zeilenerzeugung endet sauber mit ε* = -1/n."""
    g = G.GewichtsSpiel([1, 1, 1, 1], 4)
    assert SEP.separate_knapsack(g, np.full(4, 0.25)) == []
    r = _ze(g)
    assert r.eps == pytest.approx(-0.25, abs=1e-9) and EV.core_class(r.eps, g.vN) == "inneres"
    assert SA.true_violation(g, r.x) == pytest.approx(-0.25, abs=1e-9)
    _, _, zeilen = SA.sample_lp(g, 10, Stream(1))
    assert zeilen == 4 + 10


def test_diktator_unter_vielen_ist_veto_spieler():
    g = G.GewichtsSpiel([100] + [1] * 20, SC.quote_of([100] + [1] * 20, 50))
    assert g.has_veto_player() and _ze(g).eps == pytest.approx(0.0, abs=1e-9)


def test_gleichstand_bei_offenen_standorten_nimmt_die_kleinste_nummer():
    g = G.StandortSpiel(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([5.0, 5.0]))
    assert g.open_facilities().tolist() == [0]


def test_quote_grenzen_sind_inklusive():
    w = np.full(10, 10)
    assert SC.quote_of(w, 50) == 51 and SC.quote_of(w, 90) == 91
    assert SC.quote_of(w, 99) == 100                                    # mehr als 99 % von 100 heißt 100: noch beschlussfähig
    assert G.GewichtsSpiel(w, SC.quote_of(w, 90)).vN == 1.0


def test_zufallskoalitionen_bei_zwei_spielern():
    S = SA.random_coalitions(Stream(3), 2, 20, "gleich")
    assert (S.sum(axis=1) == 1).all()
    with pytest.raises(ValueError):
        SA.random_coalitions(Stream(3), 1, 5)


def test_shapley_shubik_bei_zwei_spielern_und_diktator():
    assert PW.shapley_shubik([1, 1], 2) == pytest.approx([0.5, 0.5])
    assert PW.shapley_shubik([3, 1], 3) == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError):
        PW.shapley_shubik(np.full(100, 1000), 50000)                    # Tabelle zu groß: eine klare Meldung statt Speicherhunger


def test_dreiecks_typen_mit_wenigen_kunden():
    """Jeder Typ mindestens ein Kunde: ε* = f/3 unabhängig von den Typgrößen (die drei Typpaar-Ungleichungen summieren sich zu 2 · 2f <= 3 (f + ε))."""
    for n in (3, 4, 5, 7, 12):
        assert _ze(SC.triangle_game(n, 30.0), 3).eps == pytest.approx(10.0, abs=1e-6), n


def test_obergrenze_bei_gewichtsspiel_wird_ausgewiesen():
    g = SC.weighted_game("gleich", 30, 1, 50)
    r = RG.row_generation(g, SEP.separate_knapsack, max_schleifen=2)
    assert r.abgebrochen and r.schleifen == 2 and r.eps <= 14 / 30 + 1e-9


def test_verdikt_klassen_an_der_schwelle():
    assert EV.core_class(2e-6, 1.0) == "leer" and EV.core_class(1e-6, 1.0) == "rand" and EV.core_class(-1e-6, 1.0) == "rand" and EV.core_class(-2e-6, 1.0) == "inneres"
    assert EV.core_class(1e-4, 1000.0) == "rand" and EV.core_class(2e-3, 1000.0) == "leer"      # Schwelle relativ zu Wert(N)
