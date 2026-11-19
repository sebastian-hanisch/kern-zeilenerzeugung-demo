"""Messreihen der Demo: Rechnung der Raster (E1 bis E5), Laden der vorgerechneten Tabelle `data/messreihen.json` und Kurzläufe der App.

Die Raster sind teuer (einzelne Zellen brauchen Minuten); `tools/precompute_messreihen.py` rechnet sie vor und schreibt die Tabelle mit dem Hash der Erzeuger-Quellen.
Der Hash fällt auf, sobald sich Zufallsquelle, Spielklassen oder Szenario ändern (`tests/test_messreihen.py`). Zeiten sind Messungen der Entwicklungsmaschine und hängen von Rechner und Bibliotheksstand ab.
Alle Zufallsquellen: SplitMix64 (kzl_stream). Einheiten: Zeilen = LP-Zeilen; Koalitionswerte = ausgewertete v(S) bzw. c(S); s = Sekunden je Instanz."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

import kzl_constants as C
import kzl_lp as LP
import kzl_rowgen as RG
import kzl_sampling as SA
import kzl_scenario as SC
import kzl_separation as SEP
from kzl_stream import Stream

ROOT = Path(__file__).resolve().parent
HASH_QUELLEN = ("kzl_rng.py", "kzl_stream.py", "kzl_games.py", "kzl_scenario.py")
SEEDS4 = (1, 2, 3, 4)

# --- Raster ------------------------------------------------------------------------------------------------------------------------------
E1_GEWICHT = {"erzeuger": ("gleich", "gestuft", "pareto"), "n": (10, 20, 30, 60, 100), "quote": 50, "seeds": SEEDS4}
E1_STANDORT = {"n": (10, 15, 20, 30, 40), "m": 6, "fix": 60, "k": 5, "seeds": SEEDS4}
E2_ZELLEN = ((20, 15), (30, 10))                  # (n, Zahl der Instanzen), Standortspiel m = 6, Fixkosten 60, k = 1
E2_OBERGRENZE = 400
E2_SEED0 = 100
E3_ZELLEN = (
    ("Gewicht gleich, n = 30, Quote > 50 %", {"art": "gewicht", "erzeuger": "gleich", "n": 30, "quote": 50}),
    ("Gewicht gestuft, n = 30, Quote > 60 %", {"art": "gewicht", "erzeuger": "gestuft", "n": 30, "quote": 60}),
    ("Gewicht Pareto-Tail, n = 30, Quote > 50 %", {"art": "gewicht", "erzeuger": "pareto", "n": 30, "quote": 50}),
    ("Gewicht Pareto-Tail, n = 100, Quote > 50 %", {"art": "gewicht", "erzeuger": "pareto", "n": 100, "quote": 50}),
    ("Gewicht gleichverteilt 1 bis 100, n = 30, Quote > 30 %", {"art": "gewicht", "erzeuger": "gleichverteilt", "n": 30, "quote": 30}),
    ("Gewicht gleichverteilt 1 bis 100, n = 30, Quote > 50 %", {"art": "gewicht", "erzeuger": "gleichverteilt", "n": 30, "quote": 50}),
    ("Gewicht gleichverteilt 1 bis 100, n = 30, Quote > 70 %", {"art": "gewicht", "erzeuger": "gleichverteilt", "n": 30, "quote": 70}),
    ("Standort zufällig, n = 20, m = 6, Fixkosten 60", {"art": "standort", "n": 20, "m": 6, "fix": 60}),
    ("Dreiecks-Typen mit Zufall, n = 30, Fixkosten 30", {"art": "dreieck_zufall", "n": 30, "fix": 30}),
)
E3_METHODEN = ("ze", "s500", "s5000", "s5000g", "cl", "gezielt")
E3_METHODEN_LABEL = C.E3_LABEL
E5_ZELLEN = ((60, "mehrere offene Standorte"), (600, "ein offener Standort"))
E5_N = ((8, 40), (12, 40), (20, 16), (30, 8))     # (n, Instanzen), Standortspiel m = 6, k = 8
E5_SEED0 = 5000


def erzeuger_hash() -> str:
    """Hash der Quelltexte, aus denen die Instanzen entstehen (Zeilenenden normalisiert: Windows und Linux gleich)."""
    h = hashlib.sha256()
    for name in HASH_QUELLEN:
        h.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def laden() -> dict | None:
    pfad = ROOT / C.MESSREIHEN_DATEI
    if not pfad.exists():
        return None
    return json.loads(pfad.read_text(encoding="utf-8"))


# --- E1: Zeilen gegen n ---------------------------------------------------------------------------------------------------------------------
def _lauf(game, k: int):
    t0 = time.perf_counter()
    r = RG.row_generation(game, RG.standard_separator(game), k, verlauf=False)
    return {"zeilen": r.zeilen, "schleifen": r.schleifen, "positiv": r.n_positiv, "eps_rel": r.eps / max(1.0, game.vN),
            "s": time.perf_counter() - t0, "s_lp": r.t_lp, "s_trennung": r.t_trennung, "abgebrochen": r.abgebrochen}


def e1_gewicht(erzeuger: str, n: int, seeds=SEEDS4, quote: int = 50) -> list[dict]:
    return [_lauf(SC.weighted_game(erzeuger, n, sd, quote), 1) for sd in seeds]


def e1_standort(n: int, seeds=SEEDS4, m: int = 6, fix: float = 60.0, k: int = 5) -> list[dict]:
    return [_lauf(SC.euclid_game(sd, n, m, fix), k) for sd in seeds]


# --- E2: V8-Wiederholung mit und ohne Obergrenze -------------------------------------------------------------------------------------------------
def e2_zelle(n: int, anzahl: int, obergrenze: int | None = E2_OBERGRENZE, m: int = 6, fix: float = 60.0, seed0: int = E2_SEED0) -> dict:
    mit, ohne = [], []
    for sd in range(seed0, seed0 + anzahl):
        g = SC.euclid_game(sd, n, m, fix)
        a = RG.row_generation(g, SEP.separate_facility_enum, 1, max_schleifen=obergrenze, verlauf=False) if obergrenze else None
        t0 = time.perf_counter()
        b = RG.row_generation(g, SEP.separate_facility_enum, 1, verlauf=False)
        s = time.perf_counter() - t0
        ohne.append({"zeilen": b.zeilen, "schleifen": b.schleifen, "eps_rel": b.eps / g.vN, "s": s})
        if a is not None:
            mit.append({"zeilen": a.zeilen, "abgebrochen": a.abgebrochen, "eps_rel": a.eps / g.vN})
    return {"n": n, "anzahl": anzahl, "obergrenze": obergrenze, "mit": mit, "ohne": ohne}


# --- E3: Zufall gegen gezielt -----------------------------------------------------------------------------------------------------------------
def e3_spiel(spec: dict, seed: int):
    art = spec["art"]
    if art == "gewicht":
        return SC.weighted_game(spec["erzeuger"], spec["n"], seed, spec["quote"])
    if art == "standort":
        return SC.euclid_game(seed, spec["n"], spec["m"], spec["fix"])
    if art == "dreieck_zufall":
        return SC.triangle_random_game(seed, spec["n"], spec["fix"])
    raise ValueError(art)


def e3_seeds(basis: int, anzahl: int = 4) -> list[int]:
    """Die Seeds einer Zelle: aufeinanderfolgend ab `basis`."""
    return [basis + i for i in range(anzahl)]


def e3_methode(game, methode: str, eps_stern: float, sd: int) -> dict:
    """Ein Verfahren auf einer Instanz: Aufteilung, Zeilen, Koalitionswerte, Zeit; dazu wahre und geschätzte Verletzung und die Lücke gegen ε*."""
    game.n_eval = 0
    stream = Stream(1000 + sd)
    t0 = time.perf_counter()
    if methode == "ze":
        r = RG.row_generation(game, RG.standard_separator(game), 5, verlauf=False)
        x, zeilen = r.x, r.zeilen
    elif methode in ("s500", "s5000", "s5000g"):
        K = 500 if methode == "s500" else 5000
        _, x, zeilen = SA.sample_lp(game, K, stream, "gleich" if methode.endswith("g") else "halb")
    elif methode == "cl":
        x, _ = SA.core_lagrangian(game, C.DEFAULT_ITER, C.CL_BATCH, stream, C.DEFAULT_ETA, C.CL_DELTA)
        zeilen = 0
    elif methode == "gezielt":
        _, x, zeilen, _ = SA.targeted_lp(game, stream)
    else:
        raise ValueError(methode)
    s = time.perf_counter() - t0
    evals = game.n_eval
    wahr = SA.true_violation(game, x)
    geschaetzt = SA.estimated_violation(game, x, Stream(77 + sd))
    N = max(1.0, game.vN)
    return {"luecke": (wahr - eps_stern) / N, "zeilen": zeilen, "koalitionswerte": evals, "s": s, "unterschaetzt": (wahr - geschaetzt) / N}


def e3_zelle(spec: dict, basis: int = 10, anzahl: int = 4, methoden=E3_METHODEN) -> dict:
    seeds = e3_seeds(basis, anzahl)
    eps = {}
    for sd in seeds:
        g = e3_spiel(spec, sd)
        eps[sd] = RG.row_generation(g, RG.standard_separator(g), 5, verlauf=False).eps
    out = {"seeds": seeds, "eps_stern_rel": [eps[sd] / max(1.0, e3_spiel(spec, sd).vN) for sd in seeds], "methoden": {}}
    for m in methoden:
        out["methoden"][m] = [e3_methode(e3_spiel(spec, sd), m, eps[sd], sd) for sd in seeds]
    return out


# --- E5: Standort-Regime --------------------------------------------------------------------------------------------------------------------------
def e5_zelle(fix: float, n: int, anzahl: int, m: int = 6, seed0: int = E5_SEED0, k: int = 8) -> list[dict]:
    zeilen = []
    for sd in range(seed0, seed0 + anzahl):
        g = SC.euclid_game(sd, n, m, fix)
        r = RG.row_generation(g, SEP.separate_facility_enum, k, verlauf=False)
        zeilen.append({"offen": int(len(g.open_facilities())), "eps_rel": r.eps / g.vN, "zeilen": r.zeilen})
    return zeilen


# --- Trennung einzeln: Zeit je Aufruf -----------------------------------------------------------------------------------------------------------------
def trennung_zeiten(n: int, m: int, aufrufe: int = 8, fix: float = 60.0, seed: int = 3) -> dict:
    g = SC.euclid_game(seed, n, m, fix)
    r = RG.row_generation(g, SEP.separate_facility_enum, 5, verlauf=False)
    punkte = [r.x] + [r.x + (Stream(seed + i).block(n) - 0.5) * 0.2 * g.vN / n for i in range(aufrufe - 1)]
    gleich = 0
    t_enum = t_milp = 0.0
    for x in punkte:
        t0 = time.perf_counter()
        a = SEP.separate_facility_enum(g, x, 1)[0][1]
        t_enum += time.perf_counter() - t0
        t0 = time.perf_counter()
        b = SEP.separate_facility_milp(g, x)[0][1]
        t_milp += time.perf_counter() - t0
        gleich += abs(a - b) < 1e-6 * max(1.0, g.vN)
    return {"n": n, "m": m, "aufrufe": len(punkte), "ms_enum": 1000 * t_enum / len(punkte), "ms_milp": 1000 * t_milp / len(punkte), "gleich": int(gleich)}


# --- Auswertung der Rohzeilen ---------------------------------------------------------------------------------------------------------------------
def mittel(zeilen: list[dict], feld: str) -> float:
    return float(np.mean([z[feld] for z in zeilen]))


def streuung(zeilen: list[dict], feld: str) -> float:
    return float(np.std([z[feld] for z in zeilen]))
