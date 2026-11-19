"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import math

import kzl_evaluation as EV


def test_formatierung_deutsch_und_ohne_minus_null():
    assert EV.de(0.4666667, 3) == "0,467".replace("0,467", "0,467") and EV.de(-0.5, 2) == "−0,50"
    assert EV.de(-1e-14, 3) == "0,000" and EV.de_pct(0.4667) == "46,7 %"
    assert EV.gross(1073741822) == "1.073.741.822" and EV.gross(2 ** 100 - 2) == "1,3 · 10³⁰"
    assert EV.alle_bedingungen(30) == 2 ** 30 - 2 and EV.alle_bedingungen(1) == 0


def test_verdict_kern_leer_nennt_eps_und_anteil():
    art, text = EV.verdict_kern("gewinn", "leer", 14 / 30, 1.0, 30, veto=False)
    assert art == "warning" and "Der Kern ist leer" in text and "ε* = 0,467" in text and "46,7 %" in text and "kein Mitglied hat ein Veto" in text
    art, text = EV.verdict_kern("kosten", "leer", 10.0, 60.0, 30)
    assert art == "warning" and "ε* = 10,000" in text and "16,7 %" in text and "Veto" not in text


def test_verdict_kern_widerspruch_der_handregel_wird_genannt():
    assert "Achtung" in EV.verdict_kern("gewinn", "leer", 0.2, 1.0, 5, veto=True)[1]


def test_verdict_kern_rand_und_inneres():
    art, text = EV.verdict_kern("gewinn", "rand", 0.0, 1.0, 11, veto=True)
    assert art == "info" and "kein Inneres" in text and "es gibt ein Veto-Mitglied" in text
    art, text = EV.verdict_kern("kosten", "inneres", -23.388, 1500.0, 20)
    assert art == "success" and "Inneres" in text and "ε* = −23,388" in text and "−1,6 %" in text
    assert EV.verdict_kern("gewinn", "rand", 0.0, 1.0, 1)[0] == "info"


def test_verdict_bedingungen_drei_faelle():
    art, text = EV.verdict_bedingungen(30, 67, 30, 38)
    assert art == "success" and "1.073.741.822" in text and "**67 Zeilen** in 38 Schleifen" in text and "davon 30 (höchstens n + 1 = 31)" in text
    assert EV.verdict_bedingungen(3, 6, 3, 2)[0] == "info" and "Bei n = 3 sind es nur 6 Bedingungen" in EV.verdict_bedingungen(3, 6, 3, 2)[1]
    art, text = EV.verdict_bedingungen(30, 400, 3, 400, abgebrochen=True)
    assert art == "warning" and "Abgebrochen nach 400 Schleifen" in text.replace("**", "") and "untere Schranke" in text


def test_verdict_e1_gewicht_linear_und_nicht_linear():
    linear = {"gleich": {10: 22, 20: 40, 100: 200}, "pareto": {10: 37, 20: 60, 100: 300}}
    art, text = EV.verdict_e1_gewicht(linear)
    assert art == "success" and "etwa linear" in text and "2,0 bis 3,0 Zeilen je Spieler" in text and "22 Zeilen bei n = 10" in text
    assert "300 bei n = 100" in text and "1,3 · 10³⁰" in text
    art, text = EV.verdict_e1_gewicht({"gleich": {10: 10, 20: 40, 100: 900}})          # 2,0 -> 9,0 je Spieler
    assert art == "info" and "nicht linear" in text


def test_verdict_e1_standort_nennt_mittel_std_und_je_kunde():
    art, text = EV.verdict_e1_standort({10: (50.0, 28.0), 40: (3568.0, 1413.0)})
    assert art == "info" and "50 ± 28 bei n = 10" in text and "3.568 ± 1.413".replace(".", "") in text.replace(".", "") and "89,2 gegen 5,0 Zeilen je Kunde" in text


def test_verdict_e2_mit_und_ohne_abbruch():
    art, text = EV.verdict_e2(30, 10, 7, 396.6, 786.3, 2786, 400)
    assert art == "warning" and "7 Läufe ab" in text and "396,6" in text and "786,3" in text and "2786" in text.replace(".", "")
    art, text = EV.verdict_e2(20, 15, 0, 312.1, 312.1, 1002, 400)
    assert art == "info" and "keine Instanz" in text


def test_verdict_e3_zaehlung_und_art():
    z = {"ze": (4, 4), "s500": (0, 4), "s5000": (1, 4), "s5000g": (2, 4), "cl": (0, 4), "gezielt": (4, 4)}
    art, text = EV.verdict_e3(z)
    assert art == "warning" and "Stichproben-LP, 5 000 Koalitionen: 1 von 4" in text and "gezielte Heuristik-Trennung: 4 von 4" in text and "(1 von 4 mit 5 000 Koalitionen)" in text
    z["s5000"] = (4, 4)
    assert EV.verdict_e3(z)[0] == "info"


def test_verdict_e3_live_kein_effekt_meldet_keinen_effekt():
    gleich = [{"id": "ze", "verfahren": "Zeilenerzeugung", "luecke": 0.0}, {"id": "sample", "verfahren": "Stichproben-LP", "luecke": 0.0}, {"id": "gezielt", "verfahren": "gezielt", "luecke": 0.0}]
    art, text = EV.verdict_e3_live(gleich)
    assert art == "info" and "liegen alle Verfahren im Least Core" in text
    gemischt = [gleich[0], {"id": "sample", "verfahren": "Stichproben-LP", "luecke": 0.0712}, {"id": "gezielt", "verfahren": "gezielt", "luecke": 0.0}]
    art, text = EV.verdict_e3_live(gemischt)
    assert art == "warning" and "Stichproben-LP (7,1 %)" in text and "Im Least Core liegen: gezielt" in text


def test_verdict_e4_zaehlt_ueber_der_schwelle():
    art, text = EV.verdict_e4([0.0, 0.002, 0.0005, 0.15])
    assert art == "warning" and "in 2 von 4 Fällen" in text and "größter Abstand 0,1500" in text and "Mittel 0,0381" in text
    assert EV.verdict_e4([0.0, 0.0005])[0] == "info"


def test_verdict_e5_zaehlt_beweis_und_regime():
    zellen = {60.0: {8: [(2, 0.0, 50), (3, 1e-12, 52), (1, -0.01, 30)]}, 600.0: {8: [(1, -0.08, 16), (1, -0.08, 20)]}}
    art, text = EV.verdict_e5(zellen)
    assert art == "info" and "ε* ≥ 0 in 2 von 2 Instanzen" in text and "in 3 von 3 Instanzen ein Inneres" in text
    assert "Bei n = 8 sind es im Mittel 51 Zeilen bei mehreren offenen Standorten gegen 22 bei einem" in text
    zellen[60.0][8].append((2, -0.5, 10))                                      # ein Gegenbeispiel zum Satz würde sichtbar
    assert "in 2 von 3 Instanzen" in EV.verdict_e5(zellen)[1]
    assert "Bei n =" not in EV.verdict_e5({60.0: {8: [(2, 0.0, 5)]}})[1]       # ohne gemeinsames n kein Vergleich


def test_verdict_teilung_nennt_die_beiden_teile():
    art, text = EV.verdict_teilung((351.92, 304.36, 656.28, 11), 0.0)
    assert art == "info" and "(11 Kunden)" in text and "351,92" in text and "304,36" in text and "656,28" in text and "mindestens 0,00" in text
    assert "nur ein Standort" in EV.verdict_teilung(None, -5.0)[1]


def test_verdict_bruecke_beide_zweige():
    art, text = EV.verdict_bruecke(1 / 3, 5 / 12, 1.0, 4, im_least_core=False)
    assert art == "warning" and "nicht** im Least Core" in text and "0,4167 gegen ε* = 0,3333" in text and "Abstand 0,0833" in text and "n = 4" in text
    art, text = EV.verdict_bruecke(14 / 30, 14 / 30, 1.0, 30, im_least_core=True)
    assert art == "info" and "gleich ε* = 0,4667" in text
    assert math.isclose(5 / 12 - 1 / 3, 1 / 12)
