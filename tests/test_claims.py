"""Jede Zahl der README steht hier (Zahl -> Test, Band oder exakt; DEMO-PLAYBOOK §4, §11a).

Drei Arten: (1) von Hand herleitbare Werte (exakt, bis auf LP-Genauigkeit); (2) Werte, die diese Datei neu rechnet (Bänder, weil die LP-Ecke plattformabhängig ist);
(3) Zahlen der vorgerechneten Tabelle `data/messreihen.json`: aus der Tabelle gelesen und gegen die gespeicherten Rohzeilen gehalten (Neu-Rechnen einzelner Zellen: tests/test_messreihen.py).
Zeiten stehen nicht in den Tests (Plattform), Scratch-Läufe mit mehr als 200 Instanzen stehen nur in der README und sind dort als solche genannt."""

import pytest

import kzl_auswertung as AW
import kzl_constants as C
import kzl_evaluation as EV
import kzl_messreihen as MR
import kzl_power as PW
import kzl_rowgen as RG
import kzl_sampling as SA
import kzl_scenario as SC
import kzl_separation as SEP


def _preset(name):
    p = C.PRESETS[name]
    return EV.Settings(**{k: p[k] for k in ("vehikel", "erzeuger", "n_g", "quote", "liste", "lage", "n_s", "m", "fix", "k", "seed")})


@pytest.fixture(scope="module")
def mr():
    return MR.laden()


# --- (1) von Hand ------------------------------------------------------------------------------------------------------------------------------

def test_handbeispiel_4321_quote_6_eps_ein_drittel_und_spur():
    a = EV.analyse(_preset("Handbeispiel [4, 3, 2, 1], Quote 6"))
    assert a.rg.eps == pytest.approx(1 / 3, abs=1e-8) and a.klasse == "leer"
    assert a.rg.verlauf[0].eps == pytest.approx(-0.25, abs=1e-9)             # Start mit den 4 Einer-Zeilen: ε = -1/4
    assert sum(len(s.schnitte) for s in a.rg.verlauf) == 3 and a.rg.schleifen == 4 and a.rg.zeilen == 7
    assert a.rg.x[3] == pytest.approx(0.0, abs=1e-8)                         # Spieler 4 bekommt nichts


def test_gleiche_gewichte_n_30_eps_14_dreissigstel():
    a = EV.analyse(_preset("Gleiche Gewichte, n = 30"))
    assert a.rg.eps == pytest.approx(14 / 30, abs=1e-8) and a.klasse == "leer" and not a.game.has_veto_player()
    assert EV.de(a.rg.eps, 3) == "0,467" and EV.de_pct(a.eps_rel) == "46,7 %"
    assert 55 <= a.rg.zeilen <= 80 and a.rg.n_positiv <= 31                   # gemessen 67 Zeilen; Band wegen der LP-Ecke


def test_dreiecks_typen_n_30_eps_ein_drittel_der_fixkosten():
    a = EV.analyse(_preset("Dreiecks-Typen, n = 30"))
    assert a.rg.eps == pytest.approx(10.0, abs=1e-5) and a.game.vN == pytest.approx(60.0) and a.eps_rel == pytest.approx(1 / 6, abs=1e-6)
    assert EV.de_pct(a.eps_rel) == "16,7 %" and a.klasse == "leer"
    assert 300 <= a.rg.zeilen <= 700                                          # gemessen 492 bei einer Zeile je Schleife; Band


def test_milliarde_bedingungen_bei_30_spielern():
    assert EV.alle_bedingungen(30) == 1_073_741_822 and EV.gross(EV.alle_bedingungen(30)) == "1.073.741.822" and EV.gross(EV.alle_bedingungen(100)) == "1,3 · 10³⁰"


# --- (2) neu gerechnet, im Band ------------------------------------------------------------------------------------------------------------------

def test_veto_preset_eps_null_und_wenige_zeilen():
    a = EV.analyse(_preset("Veto-Spieler (Kern nicht leer)"))
    assert a.klasse == "rand" and a.game.has_veto_player() and abs(a.rg.eps) < 1e-9 and a.rg.x[0] == pytest.approx(1.0, abs=1e-8)
    assert a.rg.zeilen <= 20 and a.rg.zeilen >= 11                            # 11 Einer-Zeilen plus wenige; gemessen 13


def test_standort_presets_offene_standorte_und_kernklasse():
    zwei = EV.analyse(_preset("Standort, mehrere offene Standorte"))
    assert len(zwei.game.open_facilities()) == 3 and zwei.klasse == "rand" and abs(zwei.rg.eps) < 1e-6
    assert 200 <= zwei.rg.zeilen <= 500                                       # gemessen 335 bei einer Zeile je Schleife
    eins = EV.analyse(_preset("Standort, ein offener Standort"))
    assert len(eins.game.open_facilities()) == 1 and eins.klasse == "inneres"
    assert -0.02 <= eins.eps_rel <= -0.012 and eins.rg.zeilen <= 80           # gemessen -1,6 % von c(N) und 41 Zeilen
    pruef = EV.teilungs_pruefung(zwei.game)
    assert pruef is not None and pruef[0] + pruef[1] <= pruef[2] + 1e-9 and EV.teilungs_pruefung(eins.game) is None


def test_pareto_n_100_preset_hinter_dem_knopf_eps_nahe_ein_halb():
    s = _preset("Pareto-Tail, n = 100")
    assert not EV.is_cheap(s) and EV.is_cheap(_preset("Pareto-Tail, n = 30"))
    a = EV.analyse(s)
    assert 0.49 <= a.rg.eps <= 0.5 and a.klasse == "leer" and 200 <= a.rg.zeilen <= 500 and a.rg.n_positiv <= 101


def test_brueckenzahlen_shapley_shubik_gegen_least_core_n_20():
    """Gleiche Gewichte: Shapley-Shubik = 1/n liegt im Least Core (4 von 4). Gestuft und Pareto-Tail (je 4 Seeds): in 0 von 8; Lücke gegen ε* zwischen 0,004 und 0,04 von Wert(N)."""
    im_lc, luecken = 0, []
    for er in ("gleich", "gestuft", "pareto"):
        for sd in (1, 2, 3, 4):
            g = SC.weighted_game(er, 20, sd, 50)
            eps = RG.row_generation(g, SEP.separate_knapsack).eps
            luecke = SA.true_violation(g, PW.shapley_shubik(g.w, g.q)) - eps
            if er == "gleich":
                im_lc += luecke < 1e-6
            else:
                luecken.append(luecke)
    assert im_lc == 4 and len(luecken) == 8 and min(luecken) > 0.004 and max(luecken) < 0.04


def test_zufaellige_standortspiele_haben_selten_einen_leeren_kern():
    """n = 10, m = 6, Fixkosten 60, 60 Seeds (1000 bis 1059): höchstens 3 leere Kerne (Scratch-Lauf: 1 von 60); die leeren Kerne kommen aus den Dreiecks-Typen und dem Gewichtsspiel."""
    leer = sum(1 for sd in range(1000, 1060) if RG.row_generation(SC.euclid_game(sd, 10, 6, 60.0), SEP.separate_facility_enum, 4, verlauf=False).eps > 1e-6 * SC.euclid_game(sd, 10, 6, 60.0).vN)
    assert leer <= 3


def test_gewichtsspiel_kern_leer_genau_ohne_veto_stichprobe():
    """Veto-Regel an 48 Instanzen (4 Erzeuger, 3 Quoten, 4 Seeds, n = 9) gegen die Rechnung; mindestens ein Fall mit nicht leerem Kern aus den Handbeispielen."""
    gezaehlt = 0
    for er in ("gleich", "gestuft", "pareto", "gleichverteilt"):
        for pct in (50, 60, 67):
            for sd in (1, 2, 3, 4):
                g = SC.weighted_game(er, 9, sd, pct)
                gezaehlt += (RG.row_generation(g, SEP.separate_knapsack).eps > 1e-6) == (not g.has_veto_player())
    assert gezaehlt == 48


# --- (3) vorgerechnete Tabelle -------------------------------------------------------------------------------------------------------------------

def test_e1_gewicht_zeilen_je_spieler_und_n_100(mr):
    tab, mittel = AW.e1_gewicht(mr)
    assert [mittel["gleich"][n] for n in (10, 20, 30, 60, 100)] == pytest.approx([22, 43, 67, 126, 219])
    assert [round(mittel["gestuft"][n], 1) for n in (10, 100)] == [28.0, 328.2] and [round(mittel["pareto"][n], 1) for n in (10, 100)] == [33.0, 323.0]
    je = [r["je_spieler"] for r in tab if r["n"] >= 20]
    assert round(min(je), 1) == 2.1 and round(max(je), 1) == 3.5
    assert all(r["positiv"] <= r["n"] + 1 for r in tab) and max(r["zeilen"] for r in tab if r["n"] == 100) < 400
    assert EV.verdict_e1_gewicht(mittel)[0] == "success"                        # "etwa linear": höchstens das Doppelte zwischen kleinstem und größtem Wert je Spieler


def test_e1_standort_wachstum_und_streuung(mr):
    tab, reihe = AW.e1_standort(mr)
    assert [round(reihe[n][0]) for n in (10, 15, 20, 30, 40)] == [76, 109, 547, 2064, 5749]
    assert [round(reihe[n][1]) for n in (10, 40)] == [18, 3127]
    assert all(2.0 <= r["positiv"] <= 3.0 for r in tab)                         # wirksame Zeilen: 2 bis 3 im Mittel, nicht n
    assert tab[-1]["s_lp"] > 100 * tab[-1]["s_trennung"] and tab[-1]["s_lp"] > tab[-1]["s_trennung"]   # die Zeit steckt im LP (Ordnung, kein fester Faktor)


def test_e2_obergrenze_verschweigt_die_zahl(mr):
    z20, z30 = AW.e2(mr)
    assert (z20["abgebrochen"], z20["anzahl"], z30["abgebrochen"], z30["anzahl"]) == (2, 15, 6, 10)
    assert (round(z20["mittel_mit"], 1), round(z20["mittel_ohne"], 1), z20["max_ohne"]) == (225.4, 229.9, 473)
    assert (round(z30["mittel_mit"], 1), round(z30["mittel_ohne"], 1), z30["max_ohne"]) == (362.2, 790.8, 3243)
    assert z30["mittel_ohne"] > 2 * z30["mittel_mit"] and z30["obergrenze"] == 400


def test_e5_regime_zaehlungen(mr):
    zellen, tab = AW.e5(mr)
    mehrere = [(e, z) for reihen in zellen.values() for l in reihen.values() for (o, e, z) in l if o >= 2]
    einer = [(e, z) for reihen in zellen.values() for l in reihen.values() for (o, e, z) in l if o == 1]
    assert (len(mehrere), len(einer)) == (88, 120) and all(e >= -1e-6 for e, _ in mehrere) and all(e < -1e-6 for e, _ in einer)
    n30 = {r["fix"]: r["zeilen"] for r in tab if r["n"] == 30}
    assert round(n30[60.0]) == 2410 or round(n30[60.0]) == 2411
    assert round(n30[600.0], 1) == 227.5


def test_trennzeiten_nur_ordnung(mr):
    assert all(z["ms_enum"] < z["ms_milp"] for z in mr["trennung"]) and len(mr["trennung"]) == 6


def test_e3_zaehlung_ueber_alle_zellen(mr):
    """Instanzen mit Lücke unter 0,001 über 9 Zellen mit je 4 Seeds (36 Instanzen)."""
    tab, zaehl, unter = AW.e3(mr)
    assert len(mr["e3"]) == 9 and zaehl == {"ze": (36, 36), "s500": (3, 36), "s5000": (7, 36), "s5000g": (4, 36), "cl": (0, 36), "gezielt": (34, 36)}


def test_e3_keine_stichprobe_ist_besser_als_die_exakte_loesung(mr):
    tab, _, _ = AW.e3(mr)
    assert all(r["luecke"] >= -1e-6 for r in tab) and all(abs(r["luecke"]) < 1e-6 for r in tab if r["verfahren"] == "ze")


def test_e3_gleiche_gewichte_ziehen_genuegt_bei_5000_nicht_bei_500(mr):
    tab, _, _ = AW.e3(mr)
    z = {r["verfahren"]: r for r in tab if r["zelle"].startswith("Gewicht gleich, n = 30")}
    assert z["s5000"]["gut"] == 4 and z["s5000g"]["gut"] == 4 and z["s500"]["gut"] == 3 and z["s500"]["luecke"] > 0.01


def test_e3_ungleiche_gewichte_5000_koalitionen_reichen_nicht_ausser_gestuft_in_3_von_4(mr):
    tab, _, _ = AW.e3(mr)
    for r in tab:
        if r["verfahren"] == "s5000" and r["zelle"].startswith("Gewicht") and not r["zelle"].startswith("Gewicht gleich,"):
            assert r["gut"] == (3 if r["zelle"].startswith("Gewicht gestuft") else 0) and 0.004 <= r["luecke"] <= 0.1, r["zelle"]


def test_e3_gezielte_suche_im_gewichtsspiel_in_allen_zellen_im_least_core(mr):
    tab, _, _ = AW.e3(mr)
    gewicht = [r for r in tab if r["verfahren"] == "gezielt" and r["zelle"].startswith("Gewicht")]
    assert len(gewicht) == 7 and all(r["gut"] == 4 and r["max_luecke"] < 1e-3 for r in gewicht)
    standort = [r for r in tab if r["verfahren"] == "gezielt" and not r["zelle"].startswith("Gewicht")]
    assert [r["gut"] for r in standort] == [3, 3] and all(r["max_luecke"] < 0.015 for r in standort)


def test_e3_core_lagrangian_nachbau_schlechter_als_stichprobe_in_8_von_9_zellen(mr):
    tab, _, _ = AW.e3(mr)
    schlechter = [z for z in mr["e3"] if next(r for r in tab if r["zelle"] == z and r["verfahren"] == "cl")["luecke"] > next(r for r in tab if r["zelle"] == z and r["verfahren"] == "s5000")["luecke"]]
    assert len(schlechter) == 8 and "Dreiecks" not in " ".join(schlechter)


def test_e3_dreiecks_typen_mehr_koalitionen_verbessern_die_luecke(mr):
    """Gegen die Ausnahme des Plans (dort verbesserten mehr Koalitionen im Dreiecks-Standortspiel nicht): hier sinkt die mittlere Lücke von 500 auf 5 000 Koalitionen."""
    tab, _, _ = AW.e3(mr)
    z = {r["verfahren"]: r["luecke"] for r in tab if r["zelle"].startswith("Dreiecks")}
    assert z["s5000"] < z["s500"] and z["s5000g"] < z["s5000"]


def test_e3_eps_stern_je_zelle(mr):
    """Mittleres ε*/Wert(N) je Zelle (Tabelle der README); Gewichtsspiele nahe der Quote, Dreiecks-Typen mit Zufall 6 bis 9 % von c(N)."""
    soll = {"Gewicht gleich, n = 30": 0.467, "Gewicht gestuft": 0.394, "Gewicht Pareto-Tail, n = 30": 0.499, "Gewicht Pareto-Tail, n = 100": 0.499, "Standort zufällig": 0.0}
    for titel, z in mr["e3"].items():
        mittel = sum(z["eps_stern_rel"]) / len(z["eps_stern_rel"])
        for kopf, wert in soll.items():
            if titel.startswith(kopf):
                assert mittel == pytest.approx(wert, abs=0.002), titel
        if titel.startswith("Dreiecks"):
            assert all(0.06 <= e <= 0.087 for e in z["eps_stern_rel"])


def test_e4_stichprobe_unterschaetzt_nie_und_oft_deutlich(mr):
    _, _, unter = AW.e3(mr)
    assert min(unter) >= -1e-6 and len(unter) == 144
    assert sum(1 for u in unter if u > 1e-3) == 123 and round(sum(unter) / len(unter), 4) == 0.036 and round(max(unter), 4) == 0.2667


def test_e3_bei_500_koalitionen_ist_die_luecke_in_jeder_zelle_groesser_als_bei_5000(mr):
    tab, _, _ = AW.e3(mr)
    for z in mr["e3"]:
        d = {r["verfahren"]: r["luecke"] for r in tab if r["zelle"] == z}
        assert d["s500"] > d["s5000"], z
