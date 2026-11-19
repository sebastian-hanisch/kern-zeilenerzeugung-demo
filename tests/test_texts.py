"""Feste Texte: die App-Fassung steht wörtlich in der README (zwei Fassungen desselben Textes werden verglichen, nicht von Hand synchron gehalten); Wortregeln der Apptexte."""

import re
from pathlib import Path

import kzl_constants as C
import kzl_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_annahmen_grenzen_probleme_stehen_wortgleich_in_der_readme():
    assert T.ASSUMPTIONS_MARKDOWN.strip() in README


def test_grenzentabelle_und_literatur_stehen_in_der_readme():
    assert T.GRENZEN_TABELLE.strip() in README
    assert all(zeile in README for zeile in T.LITERATUR)


def test_apptexte_ohne_verbotene_woerter():
    alle = " ".join([T.EINLEITUNG, T.ERKLAERTABELLE, T.ASSUMPTIONS_MARKDOWN, T.GRENZEN_TABELLE, *T.LITERATUR, *C.PRESET_HELP.values(), *C.PRESETS])
    assert not re.search(r"\btr(ä|a)gt\b|\btragen\b|Saatwert|Schwanz", alle)


def test_erklaertabelle_nennt_jedes_benannte_verfahren_der_app():
    for begriff in ("Kern", "Least Core", "Zeilenerzeugung", "Trennungsproblem", "Stichproben-LP", "Core-Lagrangian-Nachbau", "Gezielte Heuristik", "Shapley-Shubik"):
        assert f"**{begriff}" in T.ERKLAERTABELLE, begriff
