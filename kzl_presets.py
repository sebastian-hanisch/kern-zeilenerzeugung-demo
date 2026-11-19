"""Permalink (Begrenzen, Prüfen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

from dataclasses import dataclass

import streamlit as st

import kzl_constants as C
import kzl_scenario as SC
from kzl_stream import Stream


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None   # erlaubte Werte (Auswahlfelder); sonst gilt lo/hi
    liste: bool = False            # Textfeld mit Gewichtsliste: wird mit parse_liste geprüft


SETTING_SPECS = {
    "vehikel_radio": SettingSpec("veh", str, C.DEFAULT_VEHIKEL, choices=C.VEHIKEL),
    "erzeuger_select": SettingSpec("erz", str, C.DEFAULT_ERZEUGER, choices=C.ERZEUGER),
    "n_g_slider": SettingSpec("ng", int, C.DEFAULT_N_G, C.N_G_MIN, C.N_G_MAX),
    "quote_slider": SettingSpec("q", int, C.DEFAULT_QUOTE, C.QUOTE_MIN, C.QUOTE_MAX),
    "liste_input": SettingSpec("w", str, C.DEFAULT_LISTE, liste=True),
    "lage_select": SettingSpec("lage", str, C.DEFAULT_LAGE, choices=C.LAGEN),
    "n_s_slider": SettingSpec("ns", int, C.DEFAULT_N_S, C.N_S_MIN, C.N_S_MAX),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "fix_slider": SettingSpec("f", int, C.DEFAULT_FIX, C.FIX_MIN, C.FIX_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "vehikel": "vehikel_radio", "erzeuger": "erzeuger_select", "n_g": "n_g_slider", "quote": "quote_slider", "liste": "liste_input",
    "lage": "lage_select", "n_s": "n_s_slider", "m": "m_slider", "fix": "fix_slider", "k": "k_slider", "seed": "seed_input",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def parse_permalink_value(spec: SettingSpec, roh):
    """Wert aus der Adresszeile: auf die Reglergrenzen begrenzt bzw. gegen die erlaubten Werte geprüft; None heißt ungültig."""
    try:
        value = spec.caster(roh)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        return value if value in spec.choices else None
    if spec.liste:
        return value if SC.parse_liste(value) is not None else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt; Ungültiges bleibt beim Standard."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = parse_permalink_value(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos, jeder Regler aus SETTING_SPECS wird verglichen)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    """Neuer Zufalls-Seed der Instanz, ohne `random`: aus dem Zähler der Sitzung über SplitMix64 abgeleitet."""
    zaehler = st.session_state.get("seed_zaehler", 0) + 1
    st.session_state["seed_zaehler"] = zaehler
    basis = st.session_state.get("seed_input", C.DEFAULT_SEED)
    st.session_state["seed_input"] = int(Stream(basis * 1000003 + zaehler).block(1)[0] * (C.SEED_MAX + 1))
