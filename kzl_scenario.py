"""Szenario: aus den Einstellungen die Spiele erzeugen (Gewichte, Lagen, Dreiecks-Typen). Aller Zufall über `Stream` (SplitMix64, kzl_stream)."""

import numpy as np

import kzl_constants as C
import kzl_games as G
from kzl_stream import Stream


def parse_liste(text: str):
    """Gewichtsliste aus "4,3,2,1": ganze Zahlen von 1 bis LISTE_MAX_GEWICHT, 2 bis LISTE_MAX_LEN Einträge; sonst None."""
    try:
        werte = [int(t) for t in str(text).replace(";", ",").split(",") if t.strip()]
    except ValueError:
        return None
    if not (2 <= len(werte) <= C.LISTE_MAX_LEN) or any(not 1 <= v <= C.LISTE_MAX_GEWICHT for v in werte):
        return None
    return werte


def weights(erzeuger: str, n: int, seed: int, liste: str = C.DEFAULT_LISTE) -> np.ndarray:
    """Ganzzahlige Gewichte (Fuhrpark): gleich (alle 10), gestuft (5, 10, 20, 40), pareto (10 mal Pareto mit Index 2, gerundet, mindestens 1),
    gleichverteilt (1 bis 100) oder die eigene Liste (n und Seed gelten dann nicht)."""
    if erzeuger == "liste":
        werte = parse_liste(liste)
        return np.array(werte if werte is not None else parse_liste(C.DEFAULT_LISTE), dtype=int)
    if erzeuger == "gleich":
        return np.full(n, 10, dtype=int)
    u = Stream(seed).block(n)
    if erzeuger == "gestuft":
        return np.array(C.GEWICHT_STUFEN, dtype=int)[np.minimum(3, np.floor(u * 4).astype(int))]
    if erzeuger == "pareto":
        return np.maximum(1, np.round(10.0 / np.sqrt(1.0 - u))).astype(int)   # Pareto(1, 2) per Inversion: (1 - u)^(-1/2)
    if erzeuger == "gleichverteilt":
        return 1 + np.minimum(99, np.floor(u * 100).astype(int))
    raise ValueError(erzeuger)


def quote_of(w, prozent: int) -> int:
    """Ganzzahlige Quote für "Beschluss bei MEHR als `prozent` % der Gewichtssumme": floor(W * prozent / 100) + 1 (exakt in ganzen Zahlen)."""
    return (int(np.sum(w)) * int(prozent)) // 100 + 1


def weighted_game(erzeuger: str, n: int, seed: int, prozent: int, liste: str = C.DEFAULT_LISTE) -> G.GewichtsSpiel:
    w = weights(erzeuger, n, seed, liste)
    return G.GewichtsSpiel(w, quote_of(w, prozent))


def euclid_game(seed: int, n: int, m: int, fix: float, deckel: float | None = None) -> G.StandortSpiel:
    """Kunden und Standorte gleichverteilt im Quadrat [0, 100]²; gleiche Fixkosten; Weg = euklidische Entfernung (optional gedeckelt: Zukaufpreis)."""
    u = Stream(seed).block(2 * n + 2 * m)
    pk = C.KOORD_MAX * u[:2 * n].reshape(n, 2)
    ps = C.KOORD_MAX * u[2 * n:].reshape(m, 2)
    dx = pk[:, None, 0] - ps[None, :, 0]
    dy = pk[:, None, 1] - ps[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    if deckel is not None:
        d = np.minimum(d, deckel)
    return G.StandortSpiel(d, np.full(m, float(fix)))


def triangle_game(n: int, fix: float) -> G.StandortSpiel:
    """Dreiecks-Typen, rein: 3 Standorte, Kunde i hat den Typ i mod 3 und erreicht die beiden anderen Standorte ohne Weg, den eigenen Typ-Standort nur mit Weg KANTEN_ENTFERNUNG.
    Dann kosten Einer und Paare verschiedener Typen je `fix`, die große Koalition 2 fix: ε* = fix / 3 (Handrechnung in den Tests)."""
    d = np.zeros((n, 3))
    d[np.arange(n), np.arange(n) % 3] = C.KANTEN_ENTFERNUNG
    return G.StandortSpiel(d, np.full(3, float(fix)))


def triangle_random_game(seed: int, n: int, fix: float) -> G.StandortSpiel:
    """Dreiecks-Typen mit Zufall: die ersten 9 Kunden (höchstens n) reihum auf die Typen, die übrigen zufällig; Wege der erreichbaren Standorte gleichverteilt in [0, 2];
    Fixkosten je Standort fix mal Zufallszahl in [0,9; 1,1]."""
    u = Stream(seed).block(n + 3 * n + 3)
    typ = np.arange(n) % 3
    rest = max(0, n - 9)
    if rest:
        typ[9:] = np.minimum(2, np.floor(u[:rest] * 3).astype(int))
    d = 2.0 * u[n:n + 3 * n].reshape(n, 3)
    d[np.arange(n), typ] = C.KANTEN_ENTFERNUNG
    return G.StandortSpiel(d, float(fix) * (0.9 + 0.2 * u[n + 3 * n:n + 3 * n + 3]))
