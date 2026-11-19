"""End-to-end Smoke-Tests via Streamlits AppTest: Standard, jedes Preset, jeder Knopf, jeder Regler an Minimum und Maximum, Permalink, Randfälle und Worst-Case am Regler-Maximum.

Ausgegraute Regler lassen sich in AppTest (wie im Browser) nicht ändern: Regler der Standortgruppe werden daher vom Preset "Standort, mehrere offene Standorte" aus geprüft."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kzl_constants as C
import kzl_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
STANDORT = "Standort, mehrere offene Standorte"
LISTE = "Handbeispiel [4, 3, 2, 1], Quote 6"


def _fresh_app(preset: str | None = None) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    if preset:
        at.sidebar.button(key=f"preset_{preset}").click().run(timeout=TIMEOUT)
        assert not at.exception, (preset, [str(e) for e in at.exception])
    return at


def _texte(at, art: str) -> str:
    return " ".join(e.value for e in getattr(at, art))


def test_app_loads_without_exception_and_shows_the_primary_result():
    at = _fresh_app()
    assert [m.label for m in at.metric][:5] == ["ε* (Preis der Stabilität)", "ε* in % von Wert(N)", "Zeilen im LP", "Schleifen", "Rechenzeit"]
    assert at.metric[0].value == "0,467"                                           # gleiche Gewichte, n = 30: 14/30
    assert "Der Kern ist leer" in _texte(at, "warning")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_names_its_core_class(name):
    at = _fresh_app(name)
    if name == "Pareto-Tail, n = 100":                                             # n = 100: hinter dem Knopf
        assert not at.metric and any(b.label.startswith("▶️ Rechnen") for b in at.button)
        next(b for b in at.button if b.label.startswith("▶️ Rechnen")).click().run(timeout=TIMEOUT)
        assert not at.exception
    assert at.metric, name
    erwartet = {"Standort, ein offener Standort": "hat ein Inneres", "Veto-Spieler (Kern nicht leer)": "kein Inneres", "Standort, mehrere offene Standorte": "kein Inneres"}
    zeichen = erwartet.get(name, "Der Kern ist leer")
    assert zeichen in _texte(at, "warning") + _texte(at, "info") + _texte(at, "success"), name


def test_veto_preset_sampling_and_targeted_equal_exact_and_only_the_smooth_replica_misses():
    """Beim Veto-Spieler (ε* = 0, x = Einheitsvektor) liegen Stichproben-LP und gezielte Suche im Least Core (Lücke 0); der Core-Lagrangian-Nachbau erreicht die Ecke nicht
    und wird als einziger genannt. Das Verdikt meldet also einen Effekt nur dort, wo er gemessen ist."""
    at = _fresh_app("Veto-Spieler (Kern nicht leer)")
    at.button(key="live_knopf").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = _texte(at, "warning")
    assert "Die Aufteilung liegt **nicht** im Least Core bei: Core-Lagrangian-Nachbau" in text
    assert "Im Least Core liegen: Stichproben-LP, 500 Koalitionen, gezielte Heuristik-Trennung" in text


def test_every_button_in_the_main_area_runs_on_the_default_preset():
    at = _fresh_app()
    for key in ("e1_kurz_knopf", "e2_kurz_knopf", "live_knopf"):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
    assert any("Kurzlauf" in c.value or "Obergrenze" in c.value for c in at.caption) or at.dataframe
    assert "Parameter des Laufs" in " ".join(c.value for c in at.caption)


def test_live_comparison_on_a_standort_game_and_on_dreieck():
    for name in (STANDORT, "Dreiecks-Typen, n = 30"):
        at = _fresh_app(name)
        at.button(key="live_knopf").click().run(timeout=TIMEOUT)
        assert not at.exception, name
        assert "Lücke gegen ε*" in " ".join(d.value.columns.tolist().__str__() for d in at.dataframe) or at.dataframe


REGLER = [("n_g_slider", None), ("quote_slider", None), ("n_s_slider", STANDORT), ("m_slider", STANDORT), ("fix_slider", STANDORT), ("k_slider", STANDORT)]


@pytest.mark.parametrize("state_key,preset", REGLER)
@pytest.mark.parametrize("rand", ["min", "max"])
def test_every_sidebar_slider_at_min_and_max_does_not_raise(state_key, preset, rand):
    at = _fresh_app(preset)
    lo, hi = P.bounds(state_key)
    value = lo if rand == "min" else hi
    at.sidebar.slider(key=state_key).set_value(int(value)).run(timeout=TIMEOUT)
    assert not at.exception, (state_key, rand, [str(e) for e in at.exception])
    if rand == "max" and state_key in ("n_s_slider", "m_slider"):               # n = 40 bzw. m = 12 mit n = 20: hinter dem Knopf, keine Rechnung ohne Klick
        assert any(b.label.startswith("▶️ Rechnen") for b in at.button) or at.metric


def test_all_minima_and_all_maxima_without_the_expensive_run():
    for preset, werte in ((None, {"n_g_slider": C.N_G_MIN, "quote_slider": C.QUOTE_MIN}), (None, {"n_g_slider": C.N_G_MAX, "quote_slider": C.QUOTE_MAX}),
                          (STANDORT, {"n_s_slider": C.N_S_MIN, "m_slider": C.M_MIN, "fix_slider": C.FIX_MIN, "k_slider": C.K_MIN}),
                          (STANDORT, {"n_s_slider": C.N_S_MAX, "m_slider": C.M_MAX, "fix_slider": C.FIX_MAX, "k_slider": C.K_MAX})):
        at = _fresh_app(preset)
        for key, v in werte.items():
            at.sidebar.slider(key=key).set_value(int(v))
        at.run(timeout=TIMEOUT)
        assert not at.exception, (werte, [str(e) for e in at.exception])


def test_randfaelle_n_gleich_zwei_und_eigene_liste_und_ungueltige_liste():
    at = _fresh_app()
    at.sidebar.slider(key="n_g_slider").set_value(2).run(timeout=TIMEOUT)
    assert at.metric[0].value == "−0,500" and "hat ein Inneres" in _texte(at, "success")        # zwei gleiche Mitglieder, beide mit Veto
    at = _fresh_app(LISTE)
    at.sidebar.text_input(key="liste_input").set_value("kaputt").run(timeout=TIMEOUT)
    assert not at.exception and "Ungültige Liste" in _texte(at, "warning") and at.metric[0].value == "0,333"
    at.sidebar.text_input(key="liste_input").set_value("1,1").run(timeout=TIMEOUT)               # Quote 2 bei Summe 2: beide mit Veto
    assert not at.exception and at.metric[0].value == "−0,500"


def test_neue_instanz_knopf_aendert_den_seed():
    at = _fresh_app()
    vorher = at.sidebar.number_input(key="seed_input").value
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception and at.sidebar.number_input(key="seed_input").value != vorher


def test_permalink_uebernimmt_werte_begrenzt_und_prueft():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["veh"] = "standort"
    at.query_params["lage"] = "dreieck"
    at.query_params["ns"] = "99"                                                  # über der Grenze: wird auf 40 begrenzt
    at.query_params["f"] = "30"
    at.query_params["erz"] = "kaputt"                                             # ungültig: bleibt beim Standard
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.radio(key="vehikel_radio").value == "standort" and at.sidebar.selectbox(key="lage_select").value == "dreieck"
    assert at.sidebar.slider(key="n_s_slider").value == C.N_S_MAX and at.sidebar.selectbox(key="erzeuger_select").value == C.DEFAULT_ERZEUGER


def test_worst_case_am_reglermaximum_gewichtsspiel_n_100_pareto():
    """Hänger und Aufblähen fangen (kein Zeitvergleich): n = 100, Pareto-Tail, Quote 90 % läuft hinter dem Knopf durch."""
    at = _fresh_app("Pareto-Tail, n = 100")
    at.sidebar.slider(key="quote_slider").set_value(C.QUOTE_MAX).run(timeout=TIMEOUT)
    next(b for b in at.button if b.label.startswith("▶️ Rechnen")).click().run(timeout=TIMEOUT)
    assert not at.exception and at.metric and at.metric[2].value
