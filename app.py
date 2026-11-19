"""Kern bei 30 Spielern: Bedingungen gezielt suchen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Teil der Linie Koalitionen und Teilen (https://sebastianhanisch.net/konzepte-koalitionen-teilen.html). Least Core per Zeilenerzeugung für Gewichts- und Standortspiele bis n = 100 bzw. 40,
im Vergleich mit zufällig gezogenen Koalitionen. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import kzl_auswertung as AW
import kzl_constants as C
import kzl_evaluation as EV
import kzl_messreihen as MR
import kzl_power as PW
import kzl_sampling as SA
import kzl_scenario as SC
import kzl_texts as T
import kzl_visualization as V
from kzl_layout import dunkles_design, kopf_css
from kzl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kzl_stream import Stream

st.set_page_config(page_title="Kern bei 30 Spielern – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return EV.analyse(settings)


@st.cache_data(show_spinner=False)
def _messreihen():
    return MR.laden()


def _tabelle(zeilen, spalten):
    """Tabelle aus Wörterbuchzeilen; `spalten`: {Überschrift: Schlüssel}; Werte sind schon formatiert."""
    st.dataframe([{k: z[v] for k, v in spalten.items()} for z in zeilen], hide_index=True, width="stretch")


def gated(settings, key, was):
    """Analyse der Einstellungen: bei billigen Fällen sofort, sonst erst auf Knopf (mit Spinner und Zeitangabe). None heißt: noch nicht gerechnet."""
    if EV.is_cheap(settings) or st.session_state.get(f"frei_{key}") == settings:
        with st.spinner(f"Zeilenerzeugung läuft ({was}) …"):
            return _analysis(settings)
    st.info(f"{was}: diese Einstellung rechnet je nach Größe einige Sekunden bis einige Minuten (die Zeit steckt im LP, nicht in der Trennung). Die Rechnung startet auf Knopfdruck.")
    if st.button(f"▶️ Rechnen ({was})", key=f"btn_{key}"):
        st.session_state[f"frei_{key}"] = settings
        with st.spinner(f"Zeilenerzeugung läuft ({was}) …"):
            return _analysis(settings)
    return None


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kzl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧱 Kern bei 30 Spielern: Bedingungen gezielt suchen")
st.markdown(T.EINLEITUNG)

with st.expander("So funktioniert die Zeilenerzeugung", expanded=True):
    st.markdown(
        """
1. **Wenig Bedingungen:** ein LP mit nur den Einer-Koalitionen bestimmt eine Aufteilung x und das kleinste ε, das diese Bedingungen erlaubt.
2. **Trennung:** unter x wird die Koalition mit dem größten Überschuss gesucht (Gewichtsspiel: Rucksack-Problem, Standortspiel: Aufzählung der Standortmengen).
3. **Nachlegen oder Schluss:** übertrifft ihr Überschuss ε, kommt sie als neue Zeile ins LP, sonst hält x **alle** 2ⁿ − 2 Bedingungen und ε ist der Least-Core-Wert ε\\* bis auf die Abbruchschwelle 10⁻⁷.
        """
    )
    st.markdown(T.ERKLAERTABELLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Spiel**")
    vehikel = st.radio("Vehikel", C.VEHIKEL, format_func=C.VEHIKEL_LABEL.get, key="vehikel_radio",
                       help="Genossenschaft: Gewinnspiel mit Stimmgewichten. Standortspiel: Kosten teilen. Die Regler der jeweils anderen Gruppe sind ausgegraut.")
    g_aktiv = vehikel == "gewicht"
    s_aktiv = not g_aktiv
    st.markdown("**Genossenschaft** (Gewichtsspiel)")
    erzeuger = st.selectbox("Gewichte", C.ERZEUGER, format_func=C.ERZEUGER_LABEL.get, key="erzeuger_select", disabled=not g_aktiv,
                            help="Wie die Stimmgewichte der Mitglieder entstehen; „eigene Liste“ nimmt die Gewichte aus dem Textfeld.")
    liste_aktiv = erzeuger == "liste"
    n_g = st.slider("Mitglieder n", *bounds("n_g_slider"), key="n_g_slider", disabled=(not g_aktiv) or liste_aktiv,
                    help="Zahl der Mitglieder. Bei „eigene Liste“ legt die Liste n fest.")
    quote = st.slider("Beschluss bei mehr als … Prozent der Gewichtssumme", *bounds("quote_slider"), key="quote_slider", disabled=not g_aktiv,
                      help="Quote = ganzzahlig abgerundete Prozent der Gewichtssumme plus 1 (exakt in ganzen Zahlen gerechnet).")
    liste = st.text_input("Eigene Gewichte (Komma getrennt)", key="liste_input", disabled=(not g_aktiv) or not liste_aktiv,
                          help="2 bis 100 ganze Zahlen von 1 bis 1000, zum Beispiel 4,3,2,1.")
    if g_aktiv and liste_aktiv and SC.parse_liste(liste) is None:
        st.warning("Ungültige Liste: es gilt 4,3,2,1.")
    st.markdown("**Standortspiel**")
    lage = st.selectbox("Lage", C.LAGEN, format_func=C.LAGE_LABEL.get, key="lage_select", disabled=not s_aktiv,
                        help="Zufällige Punkte im Quadrat oder Dreiecks-Typen: 3 Standorte, jeder Kunde erreicht zwei davon ohne Weg (leerer Kern).")
    n_s = st.slider("Kunden n", *bounds("n_s_slider"), key="n_s_slider", disabled=not s_aktiv, help="Zahl der Kunden (Spieler).")
    m = st.slider("Mögliche Standorte m", *bounds("m_slider"), key="m_slider", disabled=(not s_aktiv) or lage != "euklid",
                  help="Bei den Dreiecks-Typen sind es immer 3 Standorte. Die Trennung zählt alle 2^m − 1 Standortmengen auf.")
    fix = st.slider("Fixkosten je Standort", *bounds("fix_slider"), key="fix_slider", disabled=not s_aktiv, help="Gleiche Fixkosten für alle Standorte; Wege liegen im Quadrat [0, 100]².")
    st.markdown("**Verfahren und Zufall**")
    k = st.slider("Zeilen je Schleife", *bounds("k_slider"), key="k_slider", disabled=not s_aktiv,
                  help="Wie viele verletzte Koalitionen je Schleife nachgelegt werden. Im Gewichtsspiel liefert die Rucksack-Trennung genau eine Zeile.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

sync_query_params({
    "vehikel_radio": vehikel, "erzeuger_select": erzeuger, "n_g_slider": int(n_g), "quote_slider": int(quote), "liste_input": liste, "lage_select": lage,
    "n_s_slider": int(n_s), "m_slider": int(m), "fix_slider": int(fix), "k_slider": int(k), "seed_input": int(seed),
})

settings = EV.Settings(vehikel, erzeuger, int(n_g), int(quote), liste, lage, int(n_s), int(m), int(fix), int(k), int(seed))
mr = _messreihen()

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
a = gated(settings, "primaer", C.VEHIKEL_LABEL[vehikel])
if a is not None:
    game, rg = a.game, a.rg
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ε* (Preis der Stabilität)", EV.de(rg.eps, 3))
    c2.metric("ε* in % von Wert(N)", EV.de_pct(a.eps_rel))
    c3.metric("Zeilen im LP", EV.gross(rg.zeilen), delta=f"von {EV.gross(EV.alle_bedingungen(game.n))}", delta_color="off")
    c4.metric("Schleifen", f"{rg.schleifen}")
    c5.metric("Rechenzeit", f"{EV.de(a.sekunden, 1)} s", delta=f"LP {EV.de(rg.t_lp, 1)} s, Trennung {EV.de(rg.t_trennung, 1)} s", delta_color="off")
    veto = game.has_veto_player() if game.kind == "gewinn" else None
    kind, text = EV.verdict_kern(game.kind, a.klasse, rg.eps, game.vN, game.n, veto)
    getattr(st, kind)(text)
    if a.veto_widerspruch:
        st.error("Die Handregel (Kern leer genau dann, wenn kein Mitglied ein Veto hat) widerspricht der Rechnung: bitte melden, das darf nicht vorkommen.")
    if rg.abgebrochen:
        st.warning(f"Die Rechnung wurde nach {C.MAX_SCHLEIFEN_LIVE} Schleifen abgebrochen (Sicherung der Live-Läufe): ε ist nur eine untere Schranke von ε*.")
    links, rechts = st.columns(2)
    with links:
        if game.kind == "gewinn":
            fig = V.build_allocation(rg.x, game.w / game.w.sum(), None, "gewinn")
            st.plotly_chart(fig, width="stretch", key="main_alloc")
            st.caption(f"Aufteilung des Werts 1 aus der Zeilenerzeugung (Balken, eine von möglicherweise mehreren) gegen den Stimmanteil. Gewichtssumme {int(game.w.sum())}, Quote: mehr als {int(game.q) - 1}, also mindestens {int(game.q)}.")
        else:
            einzel = game.value_batch(np.eye(game.n, dtype=bool), count=False)
            st.plotly_chart(V.build_allocation(rg.x, None, einzel, "kosten"), width="stretch", key="main_alloc")
            st.caption(f"Kostenaufteilung aus der Zeilenerzeugung (Balken) gegen die Kosten, die der Kunde allein trüge. c(N) = {EV.de(game.vN, 2)}; offene Standorte im Optimum: {', '.join(str(int(i) + 1) for i in game.open_facilities())}.")
    with rechts:
        if len(rg.verlauf) >= 2:
            fig = V.build_eps_curve([s.zeilen_vorher for s in rg.verlauf], [s.eps for s in rg.verlauf], rg.eps)
            st.plotly_chart(fig, width="stretch", key="main_eps")
            st.caption("ε des LP gegen die Zahl der Zeilen: mit jeder Zeile steigt ε, bis keine Koalition mehr verletzt ist (ε = ε\\*).")
        else:
            st.caption("Das LP mit den Einer-Koalitionen hielt schon alle Bedingungen: es gab nichts nachzulegen.")

    # --- Kernfrage ---------------------------------------------------------------------------------------------------------------------------
    st.markdown("## ❓ Kernfrage: Wie viele der 2ⁿ − 2 Bedingungen werden gebraucht?")
    kind, text = EV.verdict_bedingungen(game.n, rg.zeilen, rg.n_positiv, rg.schleifen, rg.abgebrochen)
    getattr(st, kind)(text)

    # --- Zeile für Zeile ---------------------------------------------------------------------------------------------------------------------
    st.subheader("🔬 Experiment: Zeile für Zeile")
    st.caption("Die Schleifen der Zeilenerzeugung auf dem Spiel der Seitenleiste: welche Koalition in welcher Schleife nachgelegt wurde.")
    verlauf = rg.verlauf
    if len(verlauf) >= 2:
        runde = st.slider("Schleife", 1, len(verlauf), 1, help="Welche Schleife angezeigt wird (die letzte findet keine verletzte Koalition mehr).")
    else:
        runde = 1
    schritt = verlauf[runde - 1] if verlauf else None
    if schritt is not None:
        st.markdown(f"**Schleife {runde}:** das LP mit {EV.gross(schritt.zeilen_vorher)} Zeilen gibt ε = {EV.de(schritt.eps, 4)}.")
        if schritt.schnitte:
            zeilen_tab = [{"Mitglieder der Koalition": EV.mitglieder_text(S), "Größe": int(S.sum()), "Überschuss unter dem x dieser Schleife": EV.de(e, 4)} for S, e in schritt.schnitte[:20]]
            st.dataframe(zeilen_tab, hide_index=True, width="stretch")
            st.caption(f"Nachgelegt: {len(schritt.schnitte)} Zeile(n) (angezeigt höchstens 20). Der Überschuss liegt über ε; deshalb war die Bedingung verletzt.")
        else:
            st.success("Keine Koalition ist mehr verletzt: x hält alle Bedingungen, ε ist ε\\*.")
    if len(verlauf) >= 2:
        with st.expander("Alle Schleifen als Tabelle"):
            st.dataframe([{"Schleife": i + 1, "Zeilen vorher": s.zeilen_vorher, "ε": EV.de(s.eps, 4), "neue Zeilen": len(s.schnitte),
                           "größter Überschuss": EV.de(max((e for _, e in s.schnitte), default=s.eps), 4)} for i, s in enumerate(verlauf[:300])], hide_index=True, width="stretch")
            if len(verlauf) > 300:
                st.caption(f"Die ersten 300 von {len(verlauf)} Schleifen.")

st.markdown("---")

# --- E1 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Wie wachsen die Zeilen mit n?")
if mr is None:
    st.warning("Die vorgerechnete Messreihe `data/messreihen.json` fehlt.")
else:
    tab_g, mittel_g = AW.e1_gewicht(mr)
    tab_s, reihe_s = AW.e1_standort(mr)
    st.caption(f"Zeilen im LP am Ende der Zeilenerzeugung, Mittel über {len(MR.SEEDS4)} Zufalls-Seeds je Zelle (vorgerechnet, Quote mehr als 50 %; Standortspiel zufällige Lage, 6 Standorte, Fixkosten 60, 5 Zeilen je Schleife).")
    c_a, c_b = st.columns(2)
    with c_a:
        st.plotly_chart(V.build_rows_vs_n({C.ERZEUGER_LABEL[er]: reihe for er, reihe in mittel_g.items()}), width="stretch", key="e1_chart_g")
    with c_b:
        st.plotly_chart(V.build_rows_vs_n({"Standortspiel": {n: v[0] for n, v in reihe_s.items()}}), width="stretch", key="e1_chart_s")
    for kind, text in (EV.verdict_e1_gewicht(mittel_g), EV.verdict_e1_standort(reihe_s)):
        getattr(st, kind)(text)
    with st.expander("Tabelle der Messreihe (Zeilen, Schleifen, wirksame Zeilen, Zeit)"):
        _tabelle([{**z, "Gewichte": C.ERZEUGER_LABEL[z["erzeuger"]], "z": f"{EV.de(z['zeilen'], 1)} ± {EV.de(z['std'], 1)}", "js": EV.de(z["je_spieler"], 2), "sch": EV.de(z["schleifen"], 0),
                   "pos": EV.de(z["positiv"], 1), "t": EV.de(z["s"], 2), "tl": EV.de(z["s_lp"], 2), "tt": EV.de(z["s_trennung"], 2)} for z in tab_g],
                 {"Gewichte": "Gewichte", "n": "n", "Zeilen (Mittel ± Std)": "z", "Zeilen je Spieler": "js", "Schleifen": "sch", "wirksame Zeilen": "pos", "s je Instanz": "t", "davon LP": "tl", "davon Trennung": "tt"})
        _tabelle([{**z, "z": f"{EV.de(z['zeilen'], 1)} ± {EV.de(z['std'], 1)}", "sch": EV.de(z["schleifen"], 0), "pos": EV.de(z["positiv"], 1), "t": EV.de(z["s"], 2),
                   "tl": EV.de(z["s_lp"], 2), "tt": EV.de(z["s_trennung"], 2)} for z in tab_s],
                 {"Standortspiel, n": "n", "Zeilen (Mittel ± Std)": "z", "Schleifen": "sch", "wirksame Zeilen": "pos", "s je Instanz": "t", "davon LP": "tl", "davon Trennung": "tt"})
        st.caption("Zeiten: Messung der Entwicklungsmaschine unter wechselnder Last, nur als Größenordnung. „wirksam“ = Zeilen mit positivem Dualwert im letzten LP.")
    with st.expander("▶️ Kurzlauf selbst rechnen (Gewichtsspiel, n = 10, 20, 30, zwei Seeds)"):
        er_kurz = st.selectbox("Gewichte", C.ERZEUGER[:4], format_func=C.ERZEUGER_LABEL.get, key="e1_kurz_erzeuger")
        if st.button("▶️ Kurzlauf starten", key="e1_kurz_knopf"):
            t0 = time.perf_counter()
            with st.spinner("Kurzlauf läuft …"):
                kurz = [{"n": n, **{"z": EV.de(MR.mittel(l, "zeilen"), 1), "sch": EV.de(MR.mittel(l, "schleifen"), 1)}, "ok": True}
                        for n in (10, 20, 30) for l in [MR.e1_gewicht(er_kurz, n, (1, 2))]]
            st.session_state["e1_kurz"] = (er_kurz, kurz, time.perf_counter() - t0)
        if "e1_kurz" in st.session_state:
            er_k, kurz, dt = st.session_state["e1_kurz"]
            _tabelle(kurz, {"n": "n", "Zeilen (Mittel, 2 Seeds)": "z", "Schleifen": "sch"})
            st.caption(f"{C.ERZEUGER_LABEL[er_k]}, Quote mehr als 50 %, {EV.de(dt, 1)} s.")

st.markdown("---")

# --- E2 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Was eine feste Obergrenze der Schleifen verschweigt")
st.caption("Standortspiel, zufällige Lage, 6 Standorte, Fixkosten 60, eine Zeile je Schleife. Dieselben Instanzen mit und ohne Obergrenze von 400 Schleifen (vorgerechnet). Die Obergrenze liefert nur eine untere Schranke von ε*, wenn sie greift.")
if mr is not None:
    e2 = AW.e2(mr)
    _tabelle([{"n": z["n"], "inst": z["anzahl"], "ab": f"{z['abgebrochen']} von {z['anzahl']}", "mit": EV.de(z["mittel_mit"], 1), "ohne": EV.de(z["mittel_ohne"], 1),
               "max": z["max_ohne"], "sch": EV.de(z["schleifen_ohne"], 1)} for z in e2],
             {"n": "n", "Instanzen": "inst", "abgebrochen mit Obergrenze": "ab", "Zeilen mit Obergrenze (Mittel)": "mit", "Zeilen ohne Obergrenze (Mittel)": "ohne", "ohne: größter Wert": "max", "ohne: Schleifen (Mittel)": "sch"})
    for z in e2:
        kind, text = EV.verdict_e2(z["n"], z["anzahl"], z["abgebrochen"], z["mittel_mit"], z["mittel_ohne"], z["max_ohne"], z["obergrenze"])
        getattr(st, kind)(text)
with st.expander("▶️ Kurzlauf: n = 20, drei Instanzen, Obergrenze nach Wahl"):
    ober = st.slider("Obergrenze der Schleifen", 50, 1000, MR.E2_OBERGRENZE, step=50, key="e2_ober")
    if st.button("▶️ Kurzlauf starten", key="e2_kurz_knopf"):
        t0 = time.perf_counter()
        with st.spinner("Kurzlauf läuft …"):
            res = MR.e2_zelle(20, 3, ober)
        st.session_state["e2_kurz"] = (ober, res, time.perf_counter() - t0)
    if "e2_kurz" in st.session_state:
        ober_k, res, dt = st.session_state["e2_kurz"]
        ab = sum(1 for r in res["mit"] if r["abgebrochen"])
        kind, text = EV.verdict_e2(20, 3, ab, MR.mittel(res["mit"], "zeilen"), MR.mittel(res["ohne"], "zeilen"), int(max(r["zeilen"] for r in res["ohne"])), ober_k)
        getattr(st, kind)(text)
        st.caption(f"Drei Instanzen (Seeds {MR.E2_SEED0} bis {MR.E2_SEED0 + 2}), {EV.de(dt, 1)} s.")

st.markdown("---")

# --- E3 und E4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Zufällig gezogene Koalitionen gegen gezielte Suche")
st.caption("Maß: Lücke = (wahre größte Verletzung der gelieferten Aufteilung − ε\\*) / Wert(N); wahre Verletzung über alle Koalitionen mit der Trennung bestimmt. Lücke 0 heißt: die Aufteilung liegt im Least Core.")
if mr is not None:
    tab3, zaehl3, unter3 = AW.e3(mr)
    titel_kurz = list(mr["e3"].keys())
    st.plotly_chart(V.build_gap_bars(titel_kurz,
                                     {m: [next(r["luecke"] for r in tab3 if r["zelle"] == t and r["verfahren"] == m) for t in titel_kurz] for m in ("s500", "s5000", "cl", "gezielt")}),
                    width="stretch", key="e3_chart")
    kind, text = EV.verdict_e3(zaehl3)
    getattr(st, kind)(text)
    with st.expander("Tabelle je Zelle und Verfahren (Mittel über die Seeds)"):
        _tabelle([{"zelle": r["zelle"], "v": C.E3_LABEL[r["verfahren"]], "l": EV.de_pct(r["luecke"], 2), "gut": f"{r['gut']} von {r['anzahl']}", "z": EV.de(r["zeilen"], 0),
                   "kw": EV.gross(int(round(r["koalitionswerte"]))), "s": EV.de(r["s"], 2)} for r in tab3],
                 {"Zelle": "zelle", "Verfahren": "v", "Lücke (Mittel)": "l", f"Lücke unter {EV.de(AW.LUECKE_SCHWELLE, 3)}": "gut", "Zeilen": "z", "Koalitionswerte": "kw", "s je Instanz": "s"})
        st.caption("Zeilen = LP-Zeilen (0 beim Core-Lagrangian-Nachbau: er löst kein LP). Koalitionswerte = ausgewertete v(S) bzw. c(S). Das Verfahren „Stichproben-LP, Größe gleichverteilt“ zieht die Größe der Koalition gleichverteilt.")

st.subheader("🔬 Experiment: Wie gut schätzt eine Stichprobe die Verletzung?")
st.caption(f"Wer den Fehler einer Aufteilung als größten Überschuss über {EV.gross(C.FEHLER_KOALITIONEN)} Zufallskoalitionen schätzt, unterschätzt ihn. Gemessen an den Aufteilungen der Stichproben-Verfahren und des Nachbaus in den Zellen oben (Abstand zwischen wahrer und geschätzter Verletzung, Anteil von Wert(N)).")
if mr is not None:
    kind, text = EV.verdict_e4(unter3)
    getattr(st, kind)(text)
    _tabelle([{"v": C.E3_LABEL[m], "mittel": EV.de(float(np.mean([r["unterschaetzt"] for z in mr["e3"].values() for r in z["methoden"][m]])), 4),
               "max": EV.de(float(np.max([r["unterschaetzt"] for z in mr["e3"].values() for r in z["methoden"][m]])), 4)} for m in ("s500", "s5000", "s5000g", "cl")],
             {"Verfahren": "v", "Unterschätzung im Mittel": "mittel", "größte Unterschätzung": "max"})

with st.expander("▶️ Live: die Verfahren auf dem Spiel der Seitenleiste vergleichen", expanded=False):
    if a is None:
        st.info("Erst die Rechnung oben starten.")
    else:
        lc1, lc2, lc3, lc4 = st.columns(4)
        K = lc1.slider("Gezogene Koalitionen K", C.STICHPROBE_MIN, C.STICHPROBE_MAX, C.DEFAULT_STICHPROBE, step=50, key="live_K")
        groesse = lc2.selectbox("Größe der Koalitionen", C.GROESSEN, format_func=C.GROESSEN_LABEL.get, key="live_groesse")
        iters = lc3.slider("Iterationen Core Lagrangian", C.ITER_MIN, C.ITER_MAX, C.DEFAULT_ITER, step=200, key="live_iter")
        eta = lc4.selectbox("Schrittweite η₀", C.ETA_WERTE, index=C.ETA_WERTE.index(C.DEFAULT_ETA), key="live_eta")
        st.caption("Zeit: der Nachbau braucht je nach Spiel Sekunden bis einige zehn Sekunden (bei m = 12 Standorten am längsten), die gezielte Suche im Standortspiel ebenso.")
        if st.button("▶️ Verfahren vergleichen", key="live_knopf"):
            game = a.game
            eps_stern = a.rg.eps
            norm = max(1.0, game.vN)
            tabelle = []
            with st.spinner("Verfahren laufen …"):
                def _zeile(vid, name, x, zeilen, evals, dt):
                    wahr = SA.true_violation(game, x)
                    schaetz = SA.estimated_violation(game, x, Stream(77 + settings.seed))
                    tabelle.append({"id": vid, "verfahren": name, "luecke": (wahr - eps_stern) / norm, "zeilen": zeilen, "koalitionswerte": evals, "s": dt, "unterschaetzt": (wahr - schaetz) / norm, "eps_rel": eps_stern / norm})
                tabelle.append({"id": "ze", "verfahren": C.E3_LABEL["ze"], "luecke": 0.0, "zeilen": a.rg.zeilen, "koalitionswerte": 0, "s": a.sekunden, "unterschaetzt": float("nan"), "eps_rel": eps_stern / norm})
                game.n_eval = 0
                t0 = time.perf_counter()
                _, x, z = SA.sample_lp(game, K, Stream(1000 + settings.seed), groesse)
                _zeile("sample", f"Stichproben-LP, {K} Koalitionen", x, z, game.n_eval, time.perf_counter() - t0)
                game.n_eval = 0
                t0 = time.perf_counter()
                x, _ = SA.core_lagrangian(game, iters, C.CL_BATCH, Stream(1000 + settings.seed), eta, C.CL_DELTA)
                _zeile("cl", "Core-Lagrangian-Nachbau", x, 0, game.n_eval, time.perf_counter() - t0)
                game.n_eval = 0
                t0 = time.perf_counter()
                _, x, z, _s = SA.targeted_lp(game, Stream(1000 + settings.seed))
                _zeile("gezielt", C.E3_LABEL["gezielt"], x, z, game.n_eval, time.perf_counter() - t0)
            st.session_state["live"] = (settings, K, groesse, iters, eta, tabelle)
        live = st.session_state.get("live")
        if live is not None and live[0] == settings:
            tabelle = live[5]
            _tabelle([{"v": r["verfahren"], "l": EV.de_pct(r["luecke"], 2), "u": "–" if r["id"] == "ze" else EV.de_pct(r["unterschaetzt"], 2), "z": EV.gross(r["zeilen"]) if r["zeilen"] else "–",
                       "kw": EV.gross(int(r["koalitionswerte"])), "s": EV.de(r["s"], 2)} for r in tabelle],
                     {"Verfahren": "v", "Lücke gegen ε*": "l", "Stichprobe unterschätzt um": "u", "Zeilen": "z", "Koalitionswerte": "kw", "s": "s"})
            kind, text = EV.verdict_e3_live(tabelle)
            getattr(st, kind)(text)
            st.caption(f"Parameter des Laufs: K = {live[1]}, Größe {live[2]}, {live[3]} Iterationen, η₀ = {live[4]}. Die Zeile der Zeilenerzeugung zeigt Zeilen und Zeit der Rechnung oben (Lücke 0 per Definition).")

st.markdown("---")

# --- E5 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Standortspiel, warum die Zeilen explodieren")
st.caption("Zufällige Lage, 6 Standorte, 8 Zeilen je Schleife. Bleibt im Optimum der großen Koalition nur ein Standort offen (hohe Fixkosten), hatte der Kern in den gemessenen Instanzen ein Inneres und es reichten wenige Zeilen; bleiben mehrere offen, liegt der Kern höchstens am Rand und die Zeilen vermehren sich (Zählungen im Verdikt unten).")
if mr is not None:
    zellen5, tab5 = AW.e5(mr)
    st.plotly_chart(V.build_regime({k_: v for k_, v in AW.e5_regime_punkte(mr).items()}), width="stretch", key="e5_chart")
    kind, text = EV.verdict_e5(zellen5)
    getattr(st, kind)(text)
    with st.expander("Tabelle je Zelle"):
        _tabelle([{"f": EV.de(r["fix"], 0), "n": r["n"], "a": r["anzahl"], "m": f"{r['mehrere']} von {r['anzahl']}", "z": EV.de(r["zeilen"], 1), "i": f"{r['mit_innerem']} von {r['anzahl']}",
                   "e": EV.de_pct(r["min_eps"], 2)} for r in tab5],
                 {"Fixkosten": "f", "n": "n", "Instanzen": "a", "mit mehreren offenen Standorten": "m", "Zeilen (Mittel)": "z", "Kern mit Innerem": "i", "kleinstes ε*/c(N)": "e"})
        st.caption("ε*/c(N) ist der Anteil von c(N); negative Werte heißen: Kern mit Innerem.")
st.markdown("**Beweisschritt am Standortspiel der Seitenleiste:** bleiben mindestens zwei Standorte offen, teilt das Optimum die Kunden in zwei Gruppen, und beide Teile zusammen kosten höchstens c(N).")
s_std = settings.als("standort")
a5 = gated(s_std, "standort", "Standortspiel der Seitenleiste")
if a5 is not None:
    kind, text = EV.verdict_teilung(EV.teilungs_pruefung(a5.game), a5.rg.eps)
    getattr(st, kind)(text)
    st.caption(f"Standortspiel der Seitenleiste (auch wenn oben das Gewichtsspiel gewählt ist): {C.LAGE_LABEL[s_std.lage]}, n = {s_std.n_s}, Fixkosten {s_std.fix}; Zeilen im LP: {EV.gross(a5.rg.zeilen)}, Zeit {EV.de(a5.sekunden, 1)} s.")

st.markdown("---")

# --- Brücke -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🌉 Brücke: Shapley-Shubik als Aufteilung")
st.markdown("Der Shapley-Shubik-Index (siehe `shapley-demo` für den Shapley-Wert) ist eine Aufteilung des Gewichtsspiels, die per Zählen statt über n! Reihenfolgen berechnet wird. Liegt sie im Least Core, also unter den Aufteilungen mit der kleinsten größten Verletzung?")
s_gew = settings.als("gewicht")
a6 = gated(s_gew, "gewicht", "Gewichtsspiel der Seitenleiste")
if a6 is not None:
    try:
        phi = PW.shapley_shubik(a6.game.w, a6.game.q)
    except ValueError as err:
        phi = None
        st.info(f"Die Zähltabelle ist für diese Gewichte zu groß: {err}.")
    if phi is not None:
        viol = SA.true_violation(a6.game, phi)
        norm = max(1.0, a6.game.vN)
        kind, text = EV.verdict_bruecke(a6.rg.eps, viol, a6.game.vN, a6.game.n, (viol - a6.rg.eps) / norm < 1e-6)
        getattr(st, kind)(text)
        st.plotly_chart(V.build_bruecke(a6.rg.x, phi), width="stretch", key="bruecke_chart")
        st.caption("Gewichtsspiel der Seitenleiste (auch wenn oben das Standortspiel gewählt ist). Die Aufteilung aus der Zeilenerzeugung ist eine von möglicherweise mehreren im Least Core; der Vergleich zeigt nur, dass Shapley-Shubik nicht dieselbe sein muss.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.GRENZEN_TABELLE)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(T.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein Spiel auf $n$ Spielern mit einer Koalitionsfunktion; gesucht sind eine Aufteilung $x$ des Gesamtwerts und der kleinste Zuschlag $\varepsilon^*$, den jede Gruppe über ihren Austrittswert hinaus höchstens verlangt.

| Symbol | Bedeutung |
|---|---|
| $N$, $S$ | alle Spieler, eine Koalition (Teilmenge) |
| $x$, $x(S)$ | Aufteilung, Summe der Anteile der Mitglieder von $S$ |
| $v(S)$, $c(S)$ | Wert einer Koalition im Gewinnspiel, Kosten im Kostenspiel |
| $w_i$, $W$, $q$ | Stimmgewicht, Summe der Gewichte, Quote |
| $f_j$, $d_{ij}$ | Fixkosten von Standort $j$, Weg von Kunde $i$ zu Standort $j$ |
| $\varepsilon^*$ | kleinster gleichmäßiger Zuschlag (Least Core) |

**Least Core, Kostenspiel.**

$$\varepsilon^* = \min_{x,\,\varepsilon}\ \varepsilon$$

$$x(S) - c(S) \le \varepsilon \quad \text{für alle } S \subsetneq N,\ S \ne \emptyset$$

$$x(N) = c(N)$$

**Least Core, Gewinnspiel.**

$$v(S) - x(S) \le \varepsilon \quad \text{für alle } S \subsetneq N,\ S \ne \emptyset, \qquad x \ge 0, \quad x(N) = v(N)$$

Der Kern ist leer genau dann, wenn $\varepsilon^* > 0$.

**Gewichtsspiel.** Mit $W = \sum_i w_i$ und dem eingestellten Prozentsatz $p$:

$$q = \left\lfloor \frac{p\,W}{100} \right\rfloor + 1 \qquad v(S) = \begin{cases} 1 & w(S) \ge q \\ 0 & \text{sonst} \end{cases}$$

Ein Mitglied $i$ hat ein Veto, wenn $W - w_i < q$ gilt; dann ist der Kern nicht leer, und ohne Veto-Mitglied ist er leer (Handregel, in der App gegen die Rechnung geprüft).

**Zeilenerzeugung.** Mit der Zeilenmenge $\mathcal{R} \subseteq 2^N$ ist das LP über $\mathcal{R}$ gelöst, ergibt $(x, \varepsilon)$, und die Trennung bestimmt

$$e^{\max} = \max_{S \notin \mathcal{R}}\ e(S, x), \qquad e(S, x) = x(S) - c(S) \ \text{ bzw. } \ v(S) - x(S)$$

Gilt $e^{\max} \le \varepsilon + \tau$ (Schwelle $\tau$ wie in den Annahmen), ist $\varepsilon = \varepsilon^*$; sonst kommt die Koalition zu $\mathcal{R}$.

**Trennung im Gewichtsspiel (Rucksack).** Mit Auswahlvariablen $y_i \in \{0,1\}$:

$$\min_{y}\ \sum_i x_i y_i \quad \text{u. d. N.} \quad \sum_i w_i y_i \ge q, \quad 1 \le \sum_i y_i \le n - 1$$

**Trennung im Standortspiel.** Zu jeder Standortmenge $F$ ist die beste Koalition $S_F = \{ i : x_i > \min_{j \in F} d_{ij} \}$ (bei leerem oder vollem $S_F$ angepasst), mit dem Überschuss

$$\max_{F}\ \Big( \sum_{i \in S_F} \big( x_i - \min_{j \in F} d_{ij} \big) - \sum_{j \in F} f_j \Big)$$

Die Demo zählt alle $2^m - 1$ nichtleeren Standortmengen $F$ auf; das MILP über $(S, F)$ liefert denselben Wert. Dieser Block ist eine Skizze: das MILP ist nur genannt, nicht ausgeschrieben.

**Shapley-Shubik-Index.**

$$\varphi_i = \sum_{S \subseteq N \setminus \{i\},\ w(S) < q \le w(S) + w_i} \frac{|S|!\,(n - |S| - 1)!}{n!}$$

Er wird über die Zahl der Teilmengen nach Größe und Gewichtssumme gezählt statt über alle Teilmengen.

**Beweisschritt zu ε\* ≥ 0 (Standortspiel).** Öffnet das Optimum von $N$ mindestens zwei Standorte, sei $S_1$ die Kundengruppe des ersten, $S_2 = N \setminus S_1$. Beide Teile des Optimums sind zulässig, also

$$c(S_1) + c(S_2) \le c(N)$$

und für jede effiziente Aufteilung gilt $e(S_1, x) + e(S_2, x) = c(N) - c(S_1) - c(S_2) \ge 0$. Mindestens eine der beiden Gruppen hat also einen Überschuss von mindestens null, und es folgt $\varepsilon^* \ge 0$.

"""
        + "\n".join([T.LITERATUR_HEAD, ""] + T.LITERATUR)
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Koalitionen und Teilen: Macht und Aufteilung](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html)."
)
