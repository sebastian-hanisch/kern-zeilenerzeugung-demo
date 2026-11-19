"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Komplexität am Regler-Maximum (DEMO-PLAYBOOK §1c):
- Gewichtsspiel, n = N_G_MAX = 100: ein LP mit höchstens n + 1 positiven Dualwerten, einige hundert Zeilen; je Schleife ein Rucksack-MILP.
  Die Zeit steckt im LP, nicht in der Trennung (Messung in der README).
- Standortspiel, m = M_MAX = 12: 2^m - 1 = 4095 Standortmengen; die Trennung zählt sie auf (O(2^m * n) je Aufruf), n <= N_S_MAX = 40.
- Core-Lagrangian-Nachbau: ITER_MAX Iterationen, je Iteration zwei Koalitionswert-Stapel der Größe CL_BATCH.
"""

# --- Regler der Seitenleiste -------------------------------------------------------------------------------------------------------------
VEHIKEL = ("gewicht", "standort")
VEHIKEL_LABEL = {"gewicht": "Genossenschaft (Gewichtsspiel, Gewinn)", "standort": "Standortspiel (Kosten)"}
DEFAULT_VEHIKEL = "gewicht"

ERZEUGER = ("gleich", "gestuft", "pareto", "gleichverteilt", "liste")
ERZEUGER_LABEL = {
    "gleich": "gleiche Gewichte (alle 10)",
    "gestuft": "gestuft (5, 10, 20 oder 40)",
    "pareto": "Pareto-Tail (10 mal Pareto mit Index 2)",
    "gleichverteilt": "gleichverteilt (1 bis 100)",
    "liste": "eigene Liste",
}
DEFAULT_ERZEUGER = "gleich"
N_G_MIN, N_G_MAX, DEFAULT_N_G = 2, 100, 30
QUOTE_MIN, QUOTE_MAX, DEFAULT_QUOTE = 50, 90, 50   # Beschluss bei mehr als diesem Prozentsatz der Gewichtssumme
LISTE_MAX_LEN = 100
LISTE_MAX_GEWICHT = 1000
DEFAULT_LISTE = "4,3,2,1"

LAGEN = ("euklid", "dreieck", "dreieck_zufall")
LAGE_LABEL = {
    "euklid": "Zufall (Punkte im Quadrat)",
    "dreieck": "Dreiecks-Typen (rein)",
    "dreieck_zufall": "Dreiecks-Typen (mit Zufall)",
}
DEFAULT_LAGE = "euklid"
N_S_MIN, N_S_MAX, DEFAULT_N_S = 5, 40, 20
M_MIN, M_MAX, DEFAULT_M = 3, 12, 6
FIX_MIN, FIX_MAX, DEFAULT_FIX = 10, 1000, 60
K_MIN, K_MAX, DEFAULT_K = 1, 8, 1                  # Zeilen je Schleife (Standortspiel)
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- feste Annahmen der Modelle ------------------------------------------------------------------------------------------------------------
TOL = 1e-7                       # Verletzungsschwelle der Zeilenerzeugung (absolut, nach Normierung auf Wert(N) = 1 bzw. in Kosteneinheiten)
EPS_TOL = 1e-6                   # |eps*| <= EPS_TOL gilt als null: Kern am Rand
KANTEN_ENTFERNUNG = 1000.0       # Weg zum "verbotenen" Standort der Dreiecks-Typen
GEWICHT_STUFEN = (5, 10, 20, 40)
KOORD_MAX = 100.0                # Kunden und Standorte liegen gleichverteilt in [0, KOORD_MAX]^2
FEHLER_KOALITIONEN = 50000       # so viele Zufallskoalitionen schätzen die Verletzung (Fehlerschätzung durch Stichprobe)

# --- Verfahren (Tab "Zufall gegen gezielt") ------------------------------------------------------------------------------------------------
STICHPROBE_MIN, STICHPROBE_MAX, DEFAULT_STICHPROBE = 50, 5000, 500
E3_LABEL = {
    "ze": "Zeilenerzeugung (exakt)", "s500": "Stichproben-LP, 500 Koalitionen", "s5000": "Stichproben-LP, 5 000 Koalitionen",
    "s5000g": "Stichproben-LP, 5 000, Größe gleichverteilt", "cl": "Core-Lagrangian-Nachbau", "gezielt": "gezielte Heuristik-Trennung",
}
GROESSEN = ("halb", "gleich")
GROESSEN_LABEL = {"halb": "jedes Mitglied mit Wahrscheinlichkeit 1/2", "gleich": "Größe gleichverteilt, Mitglieder zufällig"}
ITER_MIN, ITER_MAX, DEFAULT_ITER = 200, 4000, 2000
ETA_WERTE = (0.3, 1.0, 3.0, 10.0, 30.0)
DEFAULT_ETA = 3.0
CL_BATCH = 256
CL_DELTA = 0.01
GEZIELT_STARTS = 60
GEZIELT_BEST = 5
MAX_SCHLEIFEN_LIVE = 5000        # Sicherung der Live-Läufe; die Messreihen nutzen keine Obergrenze (Abbruch wird immer ausgewiesen)

# --- automatisch oder auf Knopf ---------------------------------------------------------------------------------------------------------------
AUTO_N_GEWICHT = 30              # Gewichtsspiel bis hierher ohne Knopf (0,3 bis 2 s)
AUTO_N_STANDORT, AUTO_M_STANDORT = 20, 10

COLOR_CHOSEN = "#2F6B65"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"
COLOR_ALT = "#d1603d"

MESSREIHEN_DATEI = "data/messreihen.json"

# --- Presets ---------------------------------------------------------------------------------------------------------------------------------
_BASIS = {
    "vehikel": DEFAULT_VEHIKEL, "erzeuger": DEFAULT_ERZEUGER, "n_g": DEFAULT_N_G, "quote": DEFAULT_QUOTE, "liste": DEFAULT_LISTE,
    "lage": DEFAULT_LAGE, "n_s": DEFAULT_N_S, "m": DEFAULT_M, "fix": DEFAULT_FIX, "k": DEFAULT_K, "seed": DEFAULT_SEED,
}


def _preset(**kw):
    return {**_BASIS, **kw}


PRESETS = {
    "Gleiche Gewichte, n = 30": _preset(),
    "Handbeispiel [4, 3, 2, 1], Quote 6": _preset(erzeuger="liste", liste="4,3,2,1"),
    "Pareto-Tail, n = 30": _preset(erzeuger="pareto"),
    "Pareto-Tail, n = 100": _preset(erzeuger="pareto", n_g=100),
    "Veto-Spieler (Kern nicht leer)": _preset(erzeuger="liste", liste="50,5,5,5,5,5,5,5,5,5,5"),
    "Dreiecks-Typen, n = 30": _preset(vehikel="standort", lage="dreieck", n_s=30, fix=30),
    "Standort, mehrere offene Standorte": _preset(vehikel="standort", n_s=20, m=6, fix=60),
    "Standort, ein offener Standort": _preset(vehikel="standort", n_s=20, m=6, fix=600),
}
PRESET_HELP = {
    "Gleiche Gewichte, n = 30": "30 Mitglieder mit gleichem Stimmgewicht, Beschluss bei mehr als der Hälfte: der Kern ist leer, die kleinste Gleichmacher-Lücke ε* hat eine Formel.",
    "Handbeispiel [4, 3, 2, 1], Quote 6": "Vier Mitglieder mit den Gewichten 4, 3, 2, 1 und Quote 6: von Hand zu lösen, ε* = 1/3.",
    "Pareto-Tail, n = 30": "30 Mitglieder, wenige sehr große Gewichte und viele kleine (Pareto-Tail): der Kern ist leer, ε* liegt nahe 1/2.",
    "Pareto-Tail, n = 100": "100 Mitglieder (Pareto-Tail): die Rechnung läuft hinter einem Knopf mit Zeitangabe.",
    "Veto-Spieler (Kern nicht leer)": "Ein Mitglied hält die Hälfte der Gewichtssumme: ohne es gibt es keinen Beschluss. Der Kern ist nicht leer, ε* = 0; kein Verfahren streitet hier.",
    "Dreiecks-Typen, n = 30": "Drei Standorte, jeder Kunde erreicht zwei davon ohne Weg: der Kern des Standortspiels ist leer, ε* = Fixkosten geteilt durch 3.",
    "Standort, mehrere offene Standorte": "Zufällige Lage, Fixkosten 60: im Optimum bleiben mehrere Standorte offen, der Kern liegt höchstens am Rand und die Zeilenerzeugung braucht mehr Zeilen als beim Preset mit einem offenen Standort.",
    "Standort, ein offener Standort": "Zufällige Lage, hohe Fixkosten 600: ein einziger Standort bleibt offen, der Kern hat Luft und wenige Zeilen reichen.",
}
