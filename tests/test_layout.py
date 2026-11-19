"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kzl_constants as C
import kzl_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"


# Regler, die in der Ausgangslage ausgegraut sind, brauchen ein Preset, das sie freischaltet (AppTest ändert keinen ausgegrauten Regler, ein Nutzer auch nicht).
START_PRESET = {"liste_input": "Handbeispiel [4, 3, 2, 1], Quote 6", **{k: "Standort, mehrere offene Standorte" for k in ("lage_select", "n_s_slider", "m_slider", "fix_slider", "k_slider")}}


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    start = START_PRESET.get(state_key, next(iter(C.PRESETS)))
    at.sidebar.button(key=f"preset_{start}").click().run()
    assert _typen(at)[start] == "primary"
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    if spec.choices is not None:
        neu = next(c for c in spec.choices if c != regler.value)
    elif spec.liste:
        neu = "7,5,3"
    else:
        neu = spec.hi if regler.value != spec.hi else spec.lo
    regler.set_value(neu).run()
    assert not at.exception
    assert _typen(at)[start] != "primary"            # die Markierung des Ausgangs-Presets ist weg (ein anderes Preset darf zufällig passen)
    at.sidebar.button(key=f"preset_{start}").click().run()   # erneutes Anklicken markiert wieder
    assert _typen(at)[start] == "primary"


def test_ausgegraute_regler_gehoeren_zum_anderen_vehikel():
    """Beim Gewichtsspiel sind die Standort-Regler ausgegraut und umgekehrt; bei eigener Liste ist n ausgegraut, sonst das Textfeld (kein stiller No-Op-Regler)."""
    at = _app()
    assert at.sidebar.slider(key="n_s_slider").disabled and at.sidebar.slider(key="m_slider").disabled and at.sidebar.slider(key="k_slider").disabled
    assert not at.sidebar.slider(key="n_g_slider").disabled and at.sidebar.text_input(key="liste_input").disabled
    at.sidebar.button(key="preset_Handbeispiel [4, 3, 2, 1], Quote 6").click().run()
    assert at.sidebar.slider(key="n_g_slider").disabled and not at.sidebar.text_input(key="liste_input").disabled
    at.sidebar.button(key="preset_Dreiecks-Typen, n = 30").click().run()
    assert at.sidebar.slider(key="n_g_slider").disabled and at.sidebar.slider(key="m_slider").disabled and not at.sidebar.slider(key="n_s_slider").disabled
    at.sidebar.button(key="preset_Standort, mehrere offene Standorte").click().run()
    assert not at.sidebar.slider(key="m_slider").disabled and not at.sidebar.slider(key="k_slider").disabled
