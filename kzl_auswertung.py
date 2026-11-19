"""Auswertung der vorgerechneten Messreihen (`data/messreihen.json`) zu Tabellen und Zählungen für App, README-Tests und Verdikte. Reine Funktionen über die geladenen Daten."""

import numpy as np

import kzl_constants as C

LUECKE_SCHWELLE = 1e-3     # Lücke gegen ε* (Anteil von Wert(N)): darunter gilt die Aufteilung als im Least Core


def _m(zeilen, feld):
    return float(np.mean([z[feld] for z in zeilen]))


def _s(zeilen, feld):
    return float(np.std([z[feld] for z in zeilen]))


def e1_gewicht(mr: dict):
    """Rückgabe: (Tabellenzeilen, {Erzeuger: {n: Mittel der Zeilen}})."""
    tabelle, mittel = [], {}
    for schluessel, zeilen in mr["e1_gewicht"].items():
        er, n = schluessel.split("|")
        n = int(n)
        mittel.setdefault(er, {})[n] = _m(zeilen, "zeilen")
        tabelle.append({"erzeuger": er, "n": n, "seeds": len(zeilen), "zeilen": _m(zeilen, "zeilen"), "std": _s(zeilen, "zeilen"), "je_spieler": _m(zeilen, "zeilen") / n,
                        "schleifen": _m(zeilen, "schleifen"), "positiv": _m(zeilen, "positiv"), "s": _m(zeilen, "s"), "s_lp": _m(zeilen, "s_lp"), "s_trennung": _m(zeilen, "s_trennung")})
    return tabelle, mittel


def e1_standort(mr: dict):
    """Rückgabe: (Tabellenzeilen, {n: (Mittel, Std der Zeilen)})."""
    tabelle, reihe = [], {}
    for n, zeilen in sorted(mr["e1_standort"].items(), key=lambda t: int(t[0])):
        n = int(n)
        reihe[n] = (_m(zeilen, "zeilen"), _s(zeilen, "zeilen"))
        tabelle.append({"n": n, "seeds": len(zeilen), "zeilen": _m(zeilen, "zeilen"), "std": _s(zeilen, "zeilen"), "schleifen": _m(zeilen, "schleifen"), "positiv": _m(zeilen, "positiv"),
                        "s": _m(zeilen, "s"), "s_lp": _m(zeilen, "s_lp"), "s_trennung": _m(zeilen, "s_trennung")})
    return tabelle, reihe


def e2(mr: dict):
    """Je n: Zahl der Instanzen, Abbrüche mit Obergrenze, Mittel der Zeilen mit und ohne, größter Wert ohne, Obergrenze."""
    out = []
    for n, z in sorted(mr["e2"].items(), key=lambda t: int(t[0])):
        mit, ohne = z["mit"], z["ohne"]
        out.append({"n": int(n), "anzahl": z["anzahl"], "obergrenze": z["obergrenze"], "abgebrochen": int(sum(1 for r in mit if r["abgebrochen"])),
                    "mittel_mit": _m(mit, "zeilen"), "mittel_ohne": _m(ohne, "zeilen"), "max_ohne": int(max(r["zeilen"] for r in ohne)),
                    "schleifen_ohne": _m(ohne, "schleifen"), "s": _m(ohne, "s")})
    return out


def e3(mr: dict):
    """Rückgabe: (Tabellenzeilen je Zelle und Verfahren, Zählung {Verfahren: (Instanzen unter der Schwelle, Instanzen)}, Liste der Unterschätzungen der Stichproben-Verfahren)."""
    tabelle, zaehlung, unter = [], {m: [0, 0] for m in C.E3_LABEL}, []
    for titel, z in mr["e3"].items():
        for m, zeilen in z["methoden"].items():
            gut = sum(1 for r in zeilen if r["luecke"] < LUECKE_SCHWELLE)
            zaehlung[m][0] += gut
            zaehlung[m][1] += len(zeilen)
            if m in ("s500", "s5000", "s5000g", "cl"):
                unter += [r["unterschaetzt"] for r in zeilen]
            tabelle.append({"zelle": titel, "verfahren": m, "luecke": _m(zeilen, "luecke"), "max_luecke": float(max(r["luecke"] for r in zeilen)), "gut": gut, "anzahl": len(zeilen),
                            "zeilen": _m(zeilen, "zeilen"), "koalitionswerte": _m(zeilen, "koalitionswerte"), "s": _m(zeilen, "s"), "unterschaetzt": _m(zeilen, "unterschaetzt")})
    return tabelle, {m: tuple(v) for m, v in zaehlung.items()}, unter


def e5(mr: dict):
    """Rückgabe: ({Fixkosten: {n: [(offen, ε*/c(N), Zeilen), ...]}}, Tabellenzeilen je Zelle)."""
    zellen, tabelle = {}, []
    for fix, z in mr["e5"].items():
        for n, liste in z["zellen"].items():
            zellen.setdefault(float(fix), {})[int(n)] = [(r["offen"], r["eps_rel"], r["zeilen"]) for r in liste]
            offen = [r["offen"] for r in liste]
            tabelle.append({"fix": float(fix), "regime": z["label"], "n": int(n), "anzahl": len(liste), "mehrere": int(sum(1 for o in offen if o >= 2)), "zeilen": _m(liste, "zeilen"),
                            "mit_innerem": int(sum(1 for r in liste if r["eps_rel"] < -C.EPS_TOL)), "min_eps": float(min(r["eps_rel"] for r in liste))})
    return zellen, tabelle


def e5_regime_punkte(mr: dict) -> dict:
    """{Beschriftung: [(n, Mittel der Zeilen), ...]} je Fixkosten."""
    out = {}
    for fix, z in mr["e5"].items():
        out[f"Fixkosten {fix}"] = [(int(n), _m(liste, "zeilen")) for n, liste in z["zellen"].items()]
    return out
