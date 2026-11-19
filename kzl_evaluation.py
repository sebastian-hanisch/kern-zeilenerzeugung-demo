"""Einstellungen, Analyse, Kennzahlen und Verdikte (reine Funktionen `verdict_*(...) -> (Art, Text)`, DEMO-PLAYBOOK §12).

Verdikte bekommen Zahlen oder Messtabellen, nie Seeds; die App ruft `getattr(st, Art)(Text)`. Zahlen im Text kommen aus der Rechnung (f-String)."""

import time
from dataclasses import dataclass, replace

import numpy as np

import kzl_constants as C
import kzl_games as G
import kzl_rowgen as RG
import kzl_scenario as SC
import kzl_separation as SEP


# --- Formatierung (deutsches Dezimalkomma, Tausenderpunkt) ----------------------------------------------------------------------------------------------
def de(x: float, stellen: int = 3) -> str:
    """Zahl mit Dezimalkomma; Minuszeichen als echtes Minus."""
    s = f"{x:.{stellen}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")                           # kein "−0,000" für Rundungsreste
    s = s.replace(".", ",")
    return s.replace("-", "−") if s.startswith("-") else s


def de_pct(anteil: float, stellen: int = 1) -> str:
    return f"{de(100.0 * anteil, stellen)} %"


def gross(k: int) -> str:
    """Ganze Zahl lesbar: bis 10^12 mit Tausenderpunkten, darüber als Zehnerpotenz mit zwei Stellen."""
    if k < 10 ** 12:
        return f"{k:,}".replace(",", ".")
    mantisse, exponent = f"{float(k):.1e}".split("e")
    hoch = str(int(exponent)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantisse.replace('.', ',')} · 10{hoch}"


def alle_bedingungen(n: int) -> int:
    """Zahl der echten nichtleeren Koalitionen: 2^n - 2."""
    return (1 << n) - 2 if n >= 2 else 0


# --- Einstellungen und Spiel ---------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    vehikel: str = C.DEFAULT_VEHIKEL
    erzeuger: str = C.DEFAULT_ERZEUGER
    n_g: int = C.DEFAULT_N_G
    quote: int = C.DEFAULT_QUOTE
    liste: str = C.DEFAULT_LISTE
    lage: str = C.DEFAULT_LAGE
    n_s: int = C.DEFAULT_N_S
    m: int = C.DEFAULT_M
    fix: int = C.DEFAULT_FIX
    k: int = C.DEFAULT_K
    seed: int = C.DEFAULT_SEED

    def als(self, vehikel: str) -> "Settings":
        return replace(self, vehikel=vehikel)


def make_game(s: Settings):
    if s.vehikel == "gewicht":
        return SC.weighted_game(s.erzeuger, s.n_g, s.seed, s.quote, s.liste)
    if s.lage == "dreieck":
        return SC.triangle_game(s.n_s, float(s.fix))
    if s.lage == "dreieck_zufall":
        return SC.triangle_random_game(s.seed, s.n_s, float(s.fix))
    return SC.euclid_game(s.seed, s.n_s, s.m, float(s.fix))


def is_cheap(s: Settings) -> bool:
    """Läuft die Analyse ohne Knopf? Gewichtsspiel bis AUTO_N_GEWICHT Spieler, Standortspiel bis n = AUTO_N_STANDORT bei m <= AUTO_M_STANDORT (Dreiecks-Typen bis n = 30)."""
    if s.vehikel == "gewicht":
        return len(SC.weights(s.erzeuger, s.n_g, s.seed, s.liste)) <= C.AUTO_N_GEWICHT
    if s.lage != "euklid":
        return s.n_s <= 30
    return s.n_s <= C.AUTO_N_STANDORT and s.m <= C.AUTO_M_STANDORT


def core_class(eps: float, wert_n: float) -> str:
    """'leer' (ε* > Schwelle), 'rand' (|ε*| <= Schwelle: nicht leer, aber ohne Inneres) oder 'inneres' (ε* < -Schwelle); Schwelle EPS_TOL * max(1, Wert(N))."""
    schwelle = C.EPS_TOL * max(1.0, abs(wert_n))
    if eps > schwelle:
        return "leer"
    return "rand" if eps >= -schwelle else "inneres"


@dataclass
class Analysis:
    settings: Settings
    game: object
    rg: RG.RowgenErgebnis
    klasse: str
    sekunden: float

    @property
    def eps_rel(self) -> float:
        return self.rg.eps / max(1.0, self.game.vN)

    @property
    def veto_widerspruch(self) -> bool:
        """Gewichtsspiel: die Handregel (Kern leer genau dann, wenn kein Veto-Spieler) widerspricht dem Ergebnis der Rechnung."""
        return self.game.kind == "gewinn" and (self.klasse == "leer") == self.game.has_veto_player()


def analyse(s: Settings) -> Analysis:
    game = make_game(s)
    t0 = time.perf_counter()
    if game.kind == "gewinn":
        rg = RG.row_generation(game, SEP.separate_knapsack, 1, max_schleifen=C.MAX_SCHLEIFEN_LIVE)
    else:
        rg = RG.row_generation(game, SEP.separate_facility_enum, s.k, max_schleifen=C.MAX_SCHLEIFEN_LIVE)
    return Analysis(s, game, rg, core_class(rg.eps, game.vN), time.perf_counter() - t0)


def mitglieder_text(maske, maximal: int = 12) -> str:
    """Nummern der Mitglieder einer Koalition (ab 1), bei langen Listen gekürzt."""
    idx = [int(i) + 1 for i in np.flatnonzero(maske)]
    if len(idx) <= maximal:
        return ", ".join(map(str, idx))
    return ", ".join(map(str, idx[:maximal])) + f", … ({len(idx)} Mitglieder)"


def teilungs_pruefung(game):
    """Beweisschritt zu 'bei mindestens zwei offenen Standorten ist ε* >= 0' (Standortspiel): S1 = Kunden des ersten offenen Standorts (nächster offener Standort);
    c(S1) + c(N ohne S1) <= c(N), weil die beiden Teile des Optimums zulässig sind. Rückgabe None bei nur einem offenen Standort, sonst (c(S1), c(Rest), c(N), |S1|)."""
    offen = game.open_facilities()
    if len(offen) < 2:
        return None
    zu = np.argmin(game.d[:, offen], axis=1)         # Gleichstand: der kleinere Standortindex
    S1 = zu == 0
    S2 = ~S1
    if not S1.any() or not S2.any():                 # ein offener Standort ohne Kunden: nur bei Fixkosten null möglich
        return None
    c1, c2 = (float(game.value_batch(S[None, :], count=False)[0]) for S in (S1, S2))
    return c1, c2, game.vN, int(S1.sum())


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------------
def verdict_kern(kind: str, klasse: str, eps: float, wert_n: float, n: int, veto: bool | None = None) -> tuple[str, str]:
    """Verdikt der Stabilität: Kern leer, am Rand oder mit Innerem; `kind` = 'gewinn' oder 'kosten'; ε* relativ zu Wert(N)."""
    rel = eps / max(1.0, abs(wert_n)) if wert_n else 0.0
    if n < 2:
        return "info", "Mit einem Spieler gibt es keine echte Koalition: der Kern besteht aus der einen Aufteilung, ε* = 0 per Definition."
    gewinn = kind == "gewinn"
    if klasse == "leer":
        satz = (f"**Der Kern ist leer.** Jede Aufteilung des Werts hat eine Gruppe, die sich um mindestens ε* = {de(eps)} "
                f"({de_pct(rel)} von Wert(N)) besser stellen würde, wenn sie austräte. Dieses ε* ist der kleinste gleichmäßige Zuschlag, der die Gruppen hält (Preis der Stabilität).")
        if veto is not None and gewinn:
            satz += " Im Gewichtsspiel passt das zur Handregel: kein Mitglied hat ein Veto." if not veto else " Achtung: die Handregel nennt hier ein Veto-Mitglied, das widerspricht der Rechnung."
        return "warning", satz
    if klasse == "rand":
        satz = ("**Der Kern ist nicht leer, hat aber kein Inneres** (ε* = 0): es gibt stabile Aufteilungen, aber jede lässt mindestens eine Gruppe genau auf ihrem Austrittswert. "
                "Rechnet man jeder echten Gruppe auch nur einen kleinen Aufschlag auf ihren Austrittswert zu, ist der Kern leer.")
        if veto is not None and gewinn and veto:
            satz += " Im Gewichtsspiel passt das zur Handregel: es gibt ein Veto-Mitglied."
        return "info", satz
    return "success", (f"**Der Kern hat ein Inneres:** ε* = {de(eps)} ({de_pct(rel)} von Wert(N)) liegt unter null, jede Gruppe kann um diesen Betrag schlechter gestellt werden und die Aufteilung bleibt stabil.")


def verdict_bedingungen(n: int, zeilen: int, positiv: int, schleifen: int, abgebrochen: bool = False) -> tuple[str, str]:
    """Verdikt der Kernfrage: wie viele der 2^n - 2 Bedingungen wurden gebraucht."""
    gesamt = alle_bedingungen(n)
    if abgebrochen:
        return "warning", (f"**Abgebrochen** nach {schleifen} Schleifen mit {gross(zeilen)} Zeilen: das Ergebnis ist nur eine untere Schranke von ε*. Die Zahl der Zeilen sagt über die Gesamtzahl nichts.")
    if zeilen >= gesamt:
        return "info", (f"Bei n = {n} sind es nur {gross(gesamt)} Bedingungen; die Zeilenerzeugung hat alle {gross(zeilen)} Zeilen gebraucht (kein Gewinn gegenüber dem vollen LP).")
    return "success", (f"Von {gross(gesamt)} Bedingungen (2^{n} − 2) kamen **{gross(zeilen)} Zeilen** in {schleifen} Schleifen ins LP, "
                       f"wirksam (positiver Dualwert) waren im letzten LP davon {positiv} (höchstens n + 1 = {n + 1}). Alle übrigen Bedingungen sind gehalten, ohne je ausgeschrieben zu werden.")


def verdict_e1_gewicht(zellen: dict) -> tuple[str, str]:
    """E1 Gewichtsspiel. `zellen`: {Erzeuger: {n: Mittel der Zeilen}} (Mittel über die Seeds der Messreihe)."""
    teile, je_spieler = [], []
    for er, reihe in zellen.items():
        ns = sorted(reihe)
        n0, n1 = ns[0], ns[-1]
        teile.append(f"{C.ERZEUGER_LABEL[er]}: {de(reihe[n0], 0)} Zeilen bei n = {n0}, {de(reihe[n1], 0)} bei n = {n1}")
        je_spieler += [reihe[n] / n for n in ns if n >= 20]
    lo, hi = min(je_spieler), max(je_spieler)
    n1 = max(max(r) for r in zellen.values())
    linear = hi <= 2.0 * lo
    kopf = (f"Im Gewichtsspiel wachsen die Zeilen mit n **etwa linear**: ab n = 20 liegen sie bei {de(lo, 1)} bis {de(hi, 1)} Zeilen je Spieler" if linear
            else f"Im Gewichtsspiel wachsen die Zeilen mit n **nicht linear**: ab n = 20 liegen sie bei {de(lo, 1)} bis {de(hi, 1)} Zeilen je Spieler")
    return ("success" if linear else "info"), f"{kopf} ({'; '.join(teile)}), gegenüber 2^{n1} − 2 ≈ {gross(alle_bedingungen(n1))} Koalitionen."


def verdict_e1_standort(reihe: dict) -> tuple[str, str]:
    """E1 Standortspiel. `reihe`: {n: (Mittel der Zeilen, Std)}."""
    ns = sorted(reihe)
    n0, n1 = ns[0], ns[-1]
    (a, sa), (b, sb) = reihe[n0], reihe[n1]
    return "info", (f"Im Standortspiel (zufällige Lage, 6 Standorte, Fixkosten 60) wachsen die Zeilen **schneller** und streuen stark: {de(a, 0)} ± {de(sa, 0)} bei n = {n0}, "
                    f"{de(b, 0)} ± {de(sb, 0)} bei n = {n1} (Mittel ± Std über die Seeds der Messreihe), also {de(b / n1, 1)} gegen {de(a / n0, 1)} Zeilen je Kunde.")


def verdict_e2(n: int, anzahl: int, abgebrochen: int, mittel_mit: float, mittel_ohne: float, max_ohne: int, obergrenze: int) -> tuple[str, str]:
    """E2 Wiederholung mit und ohne Obergrenze der Schleifen."""
    if abgebrochen == 0:
        return "info", (f"Bei n = {n} ({anzahl} Instanzen) erreichte keine Instanz die Obergrenze von {obergrenze} Schleifen: mit und ohne Obergrenze sind es {de(mittel_ohne, 1)} Zeilen im Mittel (größter Wert {max_ohne}).")
    return "warning", (f"Bei n = {n} ({anzahl} Instanzen) brach die Obergrenze von {obergrenze} Schleifen **{abgebrochen} Läufe ab**: mit Obergrenze {de(mittel_mit, 1)} Zeilen im Mittel, "
                       f"ohne {de(mittel_ohne, 1)} (größter Wert {max_ohne}). Eine feste Obergrenze verschweigt die Zahl und liefert nur eine untere Schranke von ε*.")


def verdict_e3(zaehlung: dict, schwelle: float = 1e-3) -> tuple[str, str]:
    """E3 Zufall gegen gezielt. `zaehlung`: {Verfahren: (Zahl Instanzen mit Lücke unter der Schwelle, Zahl Instanzen)}."""
    teile = [f"{C.E3_LABEL[m]}: {a} von {b}" for m, (a, b) in zaehlung.items() if m in C.E3_LABEL]
    s5, n5 = zaehlung["s5000"]
    zg, nz = zaehlung["gezielt"]
    art = "warning" if s5 < n5 else "info"
    return art, (f"Lücke gegen ε* (Anteil von Wert(N)) unter {de(schwelle, 3)} in: {'; '.join(teile)}. "
                 f"Zufällig gezogene Koalitionen liefern also nicht in allen Fällen eine Aufteilung im Least Core ({s5} von {n5} mit 5 000 Koalitionen); die gezielte Suche tat es in {zg} von {nz}, ohne Beweis.")


def verdict_e3_live(tabelle: list[dict], schwelle: float = 1e-3) -> tuple[str, str]:
    """Verdikt des Live-Vergleichs auf dem Spiel der Seitenleiste. `tabelle`: Zeilen mit 'verfahren', 'luecke' (Anteil von Wert(N))."""
    gut = [z for z in tabelle if z["id"] != "ze" and z["luecke"] < schwelle]
    schlecht = [z for z in tabelle if z["id"] != "ze" and z["luecke"] >= schwelle]
    if not schlecht:
        return "info", ("Hier liegen alle Verfahren im Least Core (Lücke unter " + de(schwelle, 3) + " von Wert(N)): in diesem Spiel hilft das gezielte Suchen nicht mehr als Ziehen. "
                        "Das ist ein Befund dieses Spiels, keine Regel.")
    namen = ", ".join(f"{z['verfahren']} ({de_pct(z['luecke'])})" for z in schlecht)
    rest = f" Im Least Core liegen: {', '.join(z['verfahren'] for z in gut)}." if gut else ""
    return "warning", f"Die Aufteilung liegt **nicht** im Least Core bei: {namen} (Lücke gegen ε* in Prozent von Wert(N)).{rest}"


def verdict_e4(unterschaetzt: list[float]) -> tuple[str, str]:
    """E4: um wie viel unterschätzt die Stichprobe von 50 000 Koalitionen die wahre Verletzung (Anteil von Wert(N)); Liste über alle Instanzen der Zellen mit Stichproben-Verfahren."""
    arr = np.asarray(unterschaetzt, dtype=float)
    mehr = int((arr > 1e-3).sum())
    return "warning" if mehr else "info", (f"Die Schätzung durch 50 000 Zufallskoalitionen lag in {mehr} von {len(arr)} Fällen um mehr als 0,001 von Wert(N) unter der wahren Verletzung "
                                           f"(Mittel {de(float(arr.mean()), 4)}, größter Abstand {de(float(arr.max()), 4)}). Sie liegt nie darüber: die Schätzung ist eine untere Schranke.")


def verdict_e5(zellen: dict) -> tuple[str, str]:
    """E5 Standort-Regime. `zellen`: {Fixkosten: {n: [(offen, eps_rel, zeilen), ...]}}. Zählt den Beweis ε* >= 0 bei mindestens zwei offenen Standorten und vergleicht die Zeilen
    beider Regime beim größten n, in dem beide vorkommen (gleiches n, nicht über alle Zellen gemittelt)."""
    mehrere, einer, je_n = [], [], {}
    for reihen in zellen.values():
        for n, liste in reihen.items():
            for o, e, z in liste:
                (mehrere if o >= 2 else einer).append(e)
                je_n.setdefault(n, {"m": [], "e": []})["m" if o >= 2 else "e"].append(z)
    ok = sum(1 for e in mehrere if e >= -C.EPS_TOL)
    inneres = sum(1 for e in einer if e < -C.EPS_TOL)
    gemeinsam = [n for n, d in je_n.items() if d["m"] and d["e"]]
    vergleich = ""
    if gemeinsam:
        n = max(gemeinsam)
        vergleich = f" Bei n = {n} sind es im Mittel {de(float(np.mean(je_n[n]['m'])), 0)} Zeilen bei mehreren offenen Standorten gegen {de(float(np.mean(je_n[n]['e'])), 0)} bei einem."
    return "info", (f"Bei mindestens zwei offenen Standorten war ε* ≥ 0 in {ok} von {len(mehrere)} Instanzen (der Kern liegt höchstens am Rand); "
                    f"bei einem offenen Standort hatte der Kern in {inneres} von {len(einer)} Instanzen ein Inneres.{vergleich}")


def verdict_teilung(pruefung, eps: float) -> tuple[str, str]:
    """Live-Beweisschritt zu ε* >= 0 bei mindestens zwei offenen Standorten."""
    if pruefung is None:
        return "info", "Im Optimum bleibt nur ein Standort offen: der Teilungsbeweis greift nicht, der Kern kann ein Inneres haben."
    c1, c2, cn, n1 = pruefung
    return "info", (f"Zwei Standorte offen: die Kunden des ersten Standorts ({n1} Kunden) kosten allein {de(c1, 2)}, die übrigen {de(c2, 2)}, zusammen {de(c1 + c2, 2)} höchstens c(N) = {de(cn, 2)}. "
                    f"Daher ist die Summe der beiden Überschüsse bei jeder effizienten Aufteilung mindestens {de(cn - c1 - c2, 2)} ≥ 0 und ε* ≥ 0 (hier ε* = {de(eps, 3)}).")


def verdict_bruecke(eps: float, verletzung_ss: float, wert_n: float, n: int, im_least_core: bool) -> tuple[str, str]:
    """Brücke: Shapley-Shubik als Aufteilung gegen den Least Core. `verletzung_ss` = größte Verletzung dieser Aufteilung (exakt)."""
    abstand = (verletzung_ss - eps) / max(1.0, wert_n)
    if im_least_core:
        return "info", (f"Die Shapley-Shubik-Aufteilung liegt hier im Least Core: ihre größte Verletzung {de(verletzung_ss, 4)} ist gleich ε* = {de(eps, 4)}.")
    return "warning", (f"Die Shapley-Shubik-Aufteilung liegt **nicht** im Least Core: größte Verletzung {de(verletzung_ss, 4)} gegen ε* = {de(eps, 4)} (Abstand {de(abstand, 4)} von Wert(N), n = {n}).")
