"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen, Prüfen), und was jedes Preset tut (Verdikt-Klasse)."""

import pytest

import kzl_constants as C
import kzl_evaluation as EV
import kzl_presets as P
import kzl_scenario as SC


def _settings(preset):
    return EV.Settings(**{k: preset[k] for k in P.PRESET_KEYS})


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            elif spec.liste:
                assert SC.parse_liste(preset[key]) is not None, name
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Gleiche Gewichte, n = 30"]) == EV.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_values_are_clamped_and_checked():
    spec = P.SETTING_SPECS
    assert P.parse_permalink_value(spec["n_g_slider"], "5000") == C.N_G_MAX
    assert P.parse_permalink_value(spec["n_g_slider"], "-3") == C.N_G_MIN
    assert P.parse_permalink_value(spec["n_g_slider"], "abc") is None
    assert P.parse_permalink_value(spec["vehikel_radio"], "standort") == "standort"
    assert P.parse_permalink_value(spec["vehikel_radio"], "boese") is None
    assert P.parse_permalink_value(spec["liste_input"], "4,3,2,1") == "4,3,2,1"
    assert P.parse_permalink_value(spec["liste_input"], "4") is None


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_has_a_core_class_matching_its_help_text(name):
    """Jedes Preset durchlaufen (Playbook 11a Punkt 7): die Klasse des Kerns stimmt mit dem überein, was die Hilfe verspricht. Die schweren Presets (n = 100) rechnet test_claims."""
    s = _settings(C.PRESETS[name])
    if not EV.is_cheap(s):
        pytest.skip("hinter dem Knopf: in test_claims")
    a = EV.analyse(s)
    erwartet = {"Gleiche Gewichte, n = 30": "leer", "Handbeispiel [4, 3, 2, 1], Quote 6": "leer", "Pareto-Tail, n = 30": "leer", "Veto-Spieler (Kern nicht leer)": "rand",
                "Dreiecks-Typen, n = 30": "leer", "Standort, mehrere offene Standorte": "rand", "Standort, ein offener Standort": "inneres"}
    assert a.klasse == erwartet[name]
    assert not a.veto_widerspruch
