"""Rechnet die Raster der Messreihen (E1 bis E5, Trennzeiten) vor und schreibt `data/messreihen.json` mit dem Hash der Erzeuger-Quellen.

Aufruf (Test-venv), Teile einzeln und parallel möglich:
    python tools/precompute_messreihen.py teil e1g            # schreibt data/teil_e1g.json
    python tools/precompute_messreihen.py teil e1s|e2|e3|e5|trennung
    python tools/precompute_messreihen.py merge               # fügt alle data/teil_*.json zu data/messreihen.json zusammen und löscht die Teile
Laufzeit: einige Minuten je Teil (E1 Standort, E2 und E3 am längsten); keine Zeitgrenze, keine Obergrenze der Schleifen außer der in E2 ausdrücklich genannten."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kzl_messreihen as M  # noqa: E402

DATA = ROOT / "data"


def teil(name: str):
    t0 = time.perf_counter()
    if name == "e1g":
        out = {f"{er}|{n}": M.e1_gewicht(er, n, M.E1_GEWICHT["seeds"], M.E1_GEWICHT["quote"])
               for er in M.E1_GEWICHT["erzeuger"] for n in M.E1_GEWICHT["n"]}
        key = "e1_gewicht"
    elif name == "e1s":
        z = M.E1_STANDORT
        out = {str(n): M.e1_standort(n, z["seeds"], z["m"], z["fix"], z["k"]) for n in z["n"]}
        key = "e1_standort"
    elif name == "e2":
        out = {str(n): M.e2_zelle(n, anzahl) for n, anzahl in M.E2_ZELLEN}
        key = "e2"
    elif name.startswith("e3"):          # "e3" = alle Zellen nacheinander, "e3:<i>" = nur Zelle i (parallel startbar)
        nummern = range(len(M.E3_ZELLEN)) if name == "e3" else [int(name.split(":")[1])]
        out = {}
        for i in nummern:
            titel, spec = M.E3_ZELLEN[i]
            out[titel] = {"spec": spec, **M.e3_zelle(spec, basis=10 + 100 * i)}
            print(f"  {titel}: {time.perf_counter() - t0:.0f} s", flush=True)
        key = "e3"
        name = name.replace(":", "_")
    elif name == "e5":
        out = {}
        for fix, label in M.E5_ZELLEN:
            out[str(fix)] = {"label": label, "zellen": {str(n): M.e5_zelle(fix, n, anzahl) for n, anzahl in M.E5_N}}
        key = "e5"
    elif name == "trennung":
        out = [M.trennung_zeiten(n, m) for n, m in ((20, 6), (20, 8), (20, 10), (30, 6), (30, 8), (30, 10))]
        key = "trennung"
    else:
        raise SystemExit(f"unbekannter Teil: {name}")
    DATA.mkdir(exist_ok=True)
    (DATA / f"teil_{name}.json").write_text(json.dumps({key: out}, ensure_ascii=False), encoding="utf-8")
    print(f"Teil {name}: {time.perf_counter() - t0:.0f} s")


def merge():
    gesamt = {"erzeuger_hash": M.erzeuger_hash()}
    teile = sorted(DATA.glob("teil_*.json"))
    for p in teile:
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            if k == "e3":
                gesamt.setdefault("e3", {}).update(v)
            else:
                gesamt[k] = v
    if "e3" in gesamt:                    # Reihenfolge wie in E3_ZELLEN
        gesamt["e3"] = {t: gesamt["e3"][t] for t, _ in M.E3_ZELLEN if t in gesamt["e3"]}
    (DATA / "messreihen.json").write_text(json.dumps(gesamt, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in teile:
        p.unlink()
    print(f"messreihen.json: {len(teile)} Teile, Hash {gesamt['erzeuger_hash']}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["teil"]:
        teil(sys.argv[2])
    elif sys.argv[1:2] == ["merge"]:
        merge()
    else:
        raise SystemExit(__doc__)
