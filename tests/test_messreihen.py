"""Vorgerechnete Messreihen (`data/messreihen.json`): Hash der Erzeuger, Aufbau, und Neu-Rechnen einzelner billiger Zellen als Band (nicht bitgleich: welche optimale Ecke HiGHS liefert,
hängt von Plattform und Bibliotheksstand ab, und damit Zeilenzahl und Schleifen; Zählungen über Instanzen, die nur an der Instanz hängen, sind exakt)."""

import pytest

import kzl_messreihen as MR
import kzl_auswertung as AW


@pytest.fixture(scope="module")
def mr():
    daten = MR.laden()
    assert daten is not None, "data/messreihen.json fehlt: tools/precompute_messreihen.py teil <name> und merge"
    return daten


def test_hash_der_erzeuger_stimmt(mr):
    """Ändert sich Zufallsquelle, Spielklasse oder Szenario, sind die Instanzen der Tabelle nicht mehr die der Demo: dann neu vorrechnen."""
    assert mr["erzeuger_hash"] == MR.erzeuger_hash()


def test_aufbau_und_zahl_der_instanzen(mr):
    assert {"e1_gewicht", "e1_standort", "e2", "e3", "e5", "trennung"} <= set(mr)
    assert set(mr["e1_gewicht"]) == {f"{er}|{n}" for er in MR.E1_GEWICHT["erzeuger"] for n in MR.E1_GEWICHT["n"]}
    assert all(len(z) == len(MR.SEEDS4) for z in mr["e1_gewicht"].values()) and all(len(z) == len(MR.SEEDS4) for z in mr["e1_standort"].values())
    assert {n: z["anzahl"] for n, z in mr["e2"].items()} == {str(n): a for n, a in MR.E2_ZELLEN}
    assert [t for t in mr["e3"]] == [t for t, _ in MR.E3_ZELLEN]
    assert all(set(z["methoden"]) == set(MR.E3_METHODEN) and len(z["seeds"]) == 4 for z in mr["e3"].values())
    assert {fix: {n: len(l) for n, l in z["zellen"].items()} for fix, z in mr["e5"].items()} == {str(f): {str(n): a for n, a in MR.E5_N} for f, _ in MR.E5_ZELLEN}


def test_keine_zelle_der_vorgerechneten_reihen_ist_abgebrochen(mr):
    assert not any(r["abgebrochen"] for z in list(mr["e1_gewicht"].values()) + list(mr["e1_standort"].values()) for r in z)


def test_neu_rechnen_billiger_zellen_im_band(mr):
    """Gleiche Instanzen (SplitMix64, auf allen Plattformen gleich), neu gerechnet: Zeilen im Band von 25 % um den gespeicherten Wert, Schleifen und Zeilen je Spieler plausibel."""
    for er, n in (("gleich", 10), ("gestuft", 10), ("pareto", 20)):
        neu = [r["zeilen"] for r in MR.e1_gewicht(er, n)]
        alt = [r["zeilen"] for r in mr["e1_gewicht"][f"{er}|{n}"]]
        assert sum(neu) == pytest.approx(sum(alt), rel=0.25), (er, n)


def test_neu_rechnen_e5_zaehlungen_exakt(mr):
    """Zahl der offenen Standorte und das Vorzeichen von ε* hängen nur an der Instanz, nicht an der LP-Ecke: exakt gleich; ε*/c(N) im Band."""
    for fix in (60, 600):
        neu = MR.e5_zelle(fix, 8, 12)
        alt = mr["e5"][str(fix)]["zellen"]["8"][:12]
        assert [r["offen"] for r in neu] == [r["offen"] for r in alt]
        assert [r["eps_rel"] < -1e-6 for r in neu] == [r["eps_rel"] < -1e-6 for r in alt]
        assert [r["eps_rel"] for r in neu] == pytest.approx([r["eps_rel"] for r in alt], abs=1e-6)


def test_neu_rechnen_e2_erste_instanz(mr):
    z = MR.e2_zelle(20, 1, MR.E2_OBERGRENZE)
    alt = mr["e2"]["20"]
    assert z["ohne"][0]["eps_rel"] == pytest.approx(alt["ohne"][0]["eps_rel"], abs=1e-6)
    assert z["ohne"][0]["zeilen"] == pytest.approx(alt["ohne"][0]["zeilen"], rel=0.3)


def test_trennzeiten_aufzaehlung_kuerzer_als_milp_und_gleiche_werte(mr):
    """Nur die Ordnung, kein Faktor (Zeitverhältnisse hängen an der Plattform); der Wert ist in allen Aufrufen gleich."""
    for z in mr["trennung"]:
        assert z["gleich"] == z["aufrufe"] == 8 and z["ms_enum"] < z["ms_milp"], z


def test_auswertungen_laufen_ueber_die_ganze_tabelle(mr):
    tab, mittel = AW.e1_gewicht(mr)
    assert len(tab) == 15 and set(mittel) == {"gleich", "gestuft", "pareto"}
    assert len(AW.e1_standort(mr)[0]) == 5 and len(AW.e2(mr)) == 2
    t3, zaehl, unter = AW.e3(mr)
    assert len(t3) == len(MR.E3_ZELLEN) * len(MR.E3_METHODEN) and all(b == 4 * len(MR.E3_ZELLEN) for _, b in zaehl.values()) and len(unter) == 4 * 4 * len(MR.E3_ZELLEN)
    zellen, tab5 = AW.e5(mr)
    assert len(tab5) == 8 and len(AW.e5_regime_punkte(mr)) == 2
