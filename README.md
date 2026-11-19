# Kern bei 30 Spielern: Bedingungen gezielt suchen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kern-zeilenerzeugung-demo.streamlit.app/)** -->

Teil der [Linie Koalitionen und Teilen](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html). Die Kante zu [`kern-demo`](https://sebastianhanisch-kern-demo.streamlit.app/) ist eine Fortsetzung über deren Grenze: `kern-demo` schreibt alle 2ⁿ − 2 Koalitionsbedingungen bis n = 10 auf, hier reichen im Gewichtsspiel wenige hundert Zeilen für n bis 100; im Standortspiel waren es bei n = 40 im Mittel 5749 Zeilen (4 Seeds), noch weit unter 2^40. Verwandt: [`shapley-demo`](https://sebastianhanisch-shapley-demo.streamlit.app/) (dort Stichproben für den Shapley-Wert, nicht für Kernbedingungen) und `machtindex-demo` (dieselbe Linie) für Machtindizes im Gewichtsspiel. Alle Stimmgewichte, Quoten, Koalitionswerte und Kosten sind Modellannahmen, keine Aussage über reale Gremien oder Betriebe.

## Kernfrage

Der Kern verlangt, dass keine der 2ⁿ − 2 Gruppen austreten will, weil sie allein besser führe. **Wie viele dieser Bedingungen werden gebraucht, wie findet man sie, und was geht schief, wenn man stattdessen zufällige Gruppen prüft?** Die App beantwortet das live mit einem Verdikt (Zahl der Zeilen gegen 2ⁿ − 2) und mit fünf Experimenten.

## Modell

Zwei Vehikel mit gleicher Schnittstelle. **Genossenschaft (Gewichtsspiel, Gewinn):** ganzzahlige Stimmgewichte, Beschluss bei mehr als einem eingestellten Prozentsatz der Gewichtssumme, v(S) = 1 für beschlussfähige Gruppen, sonst 0, Aufteilungen nichtnegativ. Der Kern ist genau dann leer, wenn kein Mitglied ein Veto hat (Handregel, in der App gegen die Rechnung geprüft). **Standortspiel (Kosten):** Kunden teilen die Fixkosten der Standorte, c(S) sind die kleinsten Kosten aus Fixkosten und Wegen. Zufällige Lage im Quadrat oder konstruierte Dreiecks-Typen (drei Standorte, jeder Kunde erreicht zwei davon ohne Weg). Gerechnet wird der Least-Core-Wert ε\*: der kleinste gleichmäßige Zuschlag, um den jede Gruppe ihren Austrittswert höchstens übertreffen darf (Kern leer ⇔ ε\* > 0). Formeln und Lesehilfe stehen im 📐-Expander der App.

## Methodik

- **Zeilenerzeugung** (`kzl_rowgen.py`): LP über die Einer-Koalitionen lösen, die am stärksten verletzte Koalition suchen (Trennung), als Zeile nachlegen, bis keine mehr verletzt ist; dann ist ε der Least-Core-Wert ε\* bis auf die Abbruchschwelle 10⁻⁷. Im Gewichtsspiel ist die Trennung ein Rucksack-MILP (`scipy.optimize.milp`, HiGHS mit voreingestellter relativer Lücke 1e-4, also nicht bis zur Schwelle 10⁻⁷ bewiesen), im Standortspiel die Aufzählung aller 2^m − 1 Standortmengen (ein MILP über Koalition und Standortmenge als Gegenprobe). Keine feste Obergrenze der Schleifen in den Messreihen; die Live-Läufe brechen nach 5 000 Schleifen ab und weisen das aus.
- **Stichproben-LP**: K zufällig gezogene Koalitionen als einzige Zeilen (jedes Mitglied mit Wahrscheinlichkeit 1/2, oder Größe gleichverteilt).
- **Core-Lagrangian-Nachbau**: Mirror Prox auf einer Lagrange-Funktion mit Stapeln zufälliger Koalitionen, nach einer Beschreibung des Verfahrens bei Gemp u. a. (2024) nachgebaut; Hyperparameter geraten bzw. abgestimmt, der Code der Autoren ist nicht gelesen.
- **Gezielte Heuristik**: LP plus Trennung durch Rucksack-Greedy (Gewicht) bzw. Einzelflip-Suche (Standort), keine Garantie.
- **Maß:** Lücke = (wahre größte Verletzung der gelieferten Aufteilung − ε\*) / Wert(N); die wahre Verletzung kommt aus der Trennung über alle Koalitionen. Lücke 0 heißt: die Aufteilung liegt im Least Core. Einheiten: Zeilen = LP-Zeilen, Koalitionswerte = ausgewertete v(S) bzw. c(S).
- **Zufall:** SplitMix64 (wörtliche Kopie `kzl_rng.py`, Kopietest), über `kzl_stream.py` als Teilströme; kein `random`, kein `numpy.random`.
- **Vorgerechnet** (`data/messreihen.json`, `tools/precompute_messreihen.py`): die Raster E1 bis E5 mit dem Hash der Erzeuger-Quellen; die App rechnet nur Kurzläufe live.

## Befunde (gemessen, keine Behauptungen)

Alle Messreihen: 4 Zufalls-Seeds je Zelle (E2: 15 bzw. 10 Instanzen, E5: 8 bis 40), Zeilen = LP-Zeilen einschließlich der n Einer-Koalitionen. Zeilen- und Schleifenzahlen hängen an der von HiGHS gelieferten LP-Ecke und sind deshalb in den Tests Bänder; die Tabelle `data/messreihen.json` ist fest, ihre Zahlen stehen exakt in `tests/test_claims.py`, billige Zellen werden in `tests/test_messreihen.py` im Band neu gerechnet.

| Frage | Befund (Voraussetzung, Zahl der Läufe) | Test |
|---|---|---|
| Liefert die Zeilenerzeugung dasselbe ε\* wie das volle LP über alle 2ⁿ − 2 Koalitionen? | Ja. Scratch-Lauf, nicht im Repo: 1 306 Vergleiche, 0 Abweichungen (Gewichtsspiel n = 11 und 13: 276 Instanzen mit vollem LP, dazu die Aufteilung gegen alle Koalitionen geprüft; Standortspiel n = 10 und 11 mit Aufzählungs- und MILP-Trennung: 320 Vergleiche; Dreiecks-Typen mit Zufall n = 11: 200; Rucksack-MILP gegen Rucksack-DP: 250; Standort-Trennung Aufzählung gegen MILP: 200; Shapley-Shubik-Zählung gegen Aufzählung: 60). Im Repo die kurze Fassung. | `tests/test_oracle_kern_zeilenerzeugung.py` |
| Handrechnungen | Gewichte 4, 3, 2, 1, Quote 6: ε\* = 1/3 bei x = (1/3, 1/3, 1/3, 0), Start ε = −1/4, in 3 Schleifen 3 Zeilen, vierte Schleife ohne Verletzung (7 Zeilen insgesamt). 30 gleiche Gewichte: ε\* = 14/30 = 0,467. Dreiecks-Typen mit Fixkosten 30: ε\* = 10 = Fixkosten/3, 16,7 % von c(N), geprüft für n = 3, 4, 5, 6, 7, 9, 12 und 30 (die Handrechnung gilt für beliebige Typgrößen). | `test_claims.py`, `test_oracle_…py`, `test_edge_cases.py` |
| Kern leer genau dann, wenn kein Veto-Mitglied (Gewichtsspiel)? | Ja in 48 von 48 Instanzen im Repo-Test (n = 9, vier Gewichtserzeuger, Quoten 50, 60, 67 %, vier Seeds); im Scratch-Lauf n = 11 (204 Instanzen, davon 202 mit leerem Kern) und n = 13 (72 Instanzen, davon 71 mit leerem Kern) ebenso; fast alle diese Instanzen haben einen leeren Kern, die Gegenrichtung (Kern nicht leer ⇒ Veto-Mitglied) ist damit nur an wenigen Fällen geprüft. Die Regel gilt für diese einfachen Spiele, die App prüft sie bei jedem Lauf. | `test_claims.py`, `test_oracle_…py` |
| Wie wachsen die Zeilen im Gewichtsspiel mit n? (E1) | Etwa linear: ab n = 20 liegen sie bei 2,1 bis 3,5 Zeilen je Spieler. Mittel über 4 Seeds, Quote mehr als 50 %, bei n = 10 / 20 / 30 / 60 / 100: gleiche Gewichte 22 / 43 / 67 / 126 / 219, gestuft 28,0 / 61,5 / 92,8 / 182,8 / 328,2, Pareto-Tail 33,0 / 69,0 / 104,2 / 199,8 / 323,0 Zeilen, gegen 2^100 − 2 ≈ 1,3 · 10³⁰ Koalitionen bei n = 100; wirksam (positiver Dualwert) höchstens n + 1 davon. | `test_claims.py::test_e1_gewicht_…` |
| … und im Standortspiel? (E1) | Schneller und mit großer Streuung (zufällige Lage, 6 Standorte, Fixkosten 60, 5 Zeilen je Schleife, 4 Seeds), Zeilen bei n = 10 / 15 / 20 / 30 / 40: 76 ± 18 / 109 ± 24 / 547 ± 299 / 2064 ± 1173 / 5749 ± 3127 (Mittel ± Std), also 143,7 bei n = 40 gegen 7,5 bei n = 10 Zeilen je Kunde; wirksam nur 2 bis 3 Zeilen im Mittel. Die Zeit steckt im LP: bei n = 40 im Mittel 104 s im LP gegen 0,2 s in der Trennung (Messung der Entwicklungsmaschine, nur Ordnung getestet). | `test_claims.py::test_e1_standort_…` |
| Was verschweigt eine feste Obergrenze von 400 Schleifen? (E2) | Standortspiel (zufällig, 6 Standorte, Fixkosten 60, eine Zeile je Schleife), dieselben Instanzen mit und ohne Obergrenze. n = 20, 15 Instanzen: 2 von 15 abgebrochen, im Mittel 225,4 Zeilen mit gegen 229,9 ohne Obergrenze (größter Wert 473). n = 30, 10 Instanzen: 6 von 10 abgebrochen, 362,2 gegen 790,8 (größter Wert 3.243). Mit Obergrenze ist ε nur eine untere Schranke von ε\*. | `test_claims.py::test_e2_…` |
| Reichen zufällig gezogene Koalitionen für den Least Core? (E3) | Nicht in allen Fällen. Instanzen mit Lücke unter 0,001 (Anteil von Wert(N)) über alle Zellen unten: Zeilenerzeugung (exakt): 36 von 36; Stichproben-LP, 500 Koalitionen: 3 von 36; Stichproben-LP, 5 000 Koalitionen: 7 von 36; Stichproben-LP, 5 000, Größe gleichverteilt: 4 von 36; Core-Lagrangian-Nachbau: 0 von 36; gezielte Heuristik-Trennung: 34 von 36. Die Tabelle zeigt je Zelle die mittlere Lücke und in Klammern die Zahl der Instanzen unter der Schwelle. Bei gleichen Gewichten genügen 5 000 gezogene Koalitionen (4 von 4), bei gestuften Gewichten reichen sie in 3 von 4, in den übrigen Gewichtszellen in 0 von 4 (mittlere Lücke 0,008 bis 0,097); bei 500 Koalitionen ist die Lücke überall größer als bei 5 000. | `test_claims.py::test_e3_…` |
| Wie gut schätzt eine Stichprobe die Verletzung? (E4) | Die Schätzung durch 50 000 Zufallskoalitionen lag in 123 von 144 Fällen um mehr als 0,001 von Wert(N) unter der wahren Verletzung (Mittel 0,0360, größter Abstand 0,2667). Sie liegt nie darüber: die Schätzung ist eine untere Schranke. | `test_claims.py::test_e4_…` |
| Wann explodieren die Zeilen im Standortspiel? (E5) | Bei mindestens zwei offenen Standorten war ε\* ≥ 0 in 88 von 88 Instanzen; bei einem offenen Standort hatte der Kern in 120 von 120 Instanzen ein Inneres (zufällige Lage, 6 Standorte, Fixkosten 60 und 600, n = 8 bis 30, 8 Zeilen je Schleife). Bei n = 30 sind es im Mittel 2410 Zeilen (Fixkosten 60, meist mehrere offene Standorte) gegen 227,5 (Fixkosten 600, ein offener Standort); Fixkosten und Zahl offener Standorte ändern sich dabei gemeinsam. Beweis von ε\* ≥ 0 bei mindestens zwei offenen Standorten im 📐-Expander; die Teilungsprüfung läuft live. | `test_claims.py::test_e5_…`, `test_claims.py::test_standort_presets_…` |
| Wie schnell sind die beiden Trennungen im Standortspiel? | Zeit je Aufruf (8 Aufrufe, n = 20 und 30, m = 6, 8, 10): Aufzählung 0,1 bis 0,4 ms, MILP 10 bis 16 ms; gleicher größter Überschuss in 48 von 48 Aufrufen. Die Zeiten sind Maschinenwerte; getestet wird nur, dass die Aufzählung in allen 6 Zellen kürzer ist. | `test_messreihen.py::test_trennzeiten_…` |
| Hat ein zufälliges Standortspiel einen leeren Kern? | Selten: n = 10, 6 Standorte, Fixkosten 60, 60 Instanzen: 1 leerer Kern (Scratch-Lauf; der Repo-Test lässt höchstens 3 zu); mit Zukaufpreis-Deckel 30: 0 von 60; n = 11, 7 Standorte, Fixkosten 40, Deckel 25: 0 von 40; n = 20, 12 Standorte, Fixkosten 200: 0 von 50 (Seeds 910 bis 959, Scratch-Lauf). Leere Kerne entstehen aus dem Gewichtsspiel und den Dreiecks-Typen. | `test_claims.py::test_zufaellige_standortspiele_…` |
| Liegt die Shapley-Shubik-Aufteilung im Least Core? (Brücke) | Bei gleichen Gewichten ja (4 von 4, n = 20). Bei gestuften Gewichten und Pareto-Tail (n = 20, je 4 Seeds): in 0 von 8; die Lücke gegen ε\* liegt zwischen 0,004 und 0,04 von Wert(N). | `test_claims.py::test_brueckenzahlen_…` |
| Läuft das Regler-Maximum durch? | Gewichtsspiel n = 100, Pareto-Tail, Quote 90 % hinter dem Knopf: ja (Worst-Case-Test ohne Zeitvergleich). Standortspiel mit n = 40 liegt hinter dem Knopf und braucht je nach Lage bis zu mehreren Minuten (E1: Mittel 104 s im LP bei 5 Zeilen je Schleife, zufällige Lage, 6 Standorte). | `tests/test_app.py` |

### Zufall gegen gezielt je Zelle (E3, Lücke gegen ε\*, Anteil von Wert(N))

Mittlere Lücke über 4 Zufalls-Seeds, in Klammern: Instanzen mit Lücke unter 0,001. Spalte ε\*/Wert(N): Mittel der Zelle (Zellen mit Standortspiel: Anteil von c(N)). Stichproben-LP mit 500 und 5 000 gezogenen Koalitionen (jedes Mitglied mit Wahrscheinlichkeit 1/2), mit 5 000 Koalitionen gleichverteilter Größe, Core-Lagrangian-Nachbau (2 000 Iterationen, Stapel 256, η₀ = 3, δ = 0,01) und gezielte Heuristik. 

| Zelle | ε\*/Wert(N) | K = 500 | K = 5 000 | K = 5 000, Größe gleichverteilt | Core Lagrangian | gezielt |
|---|---|---|---|---|---|---|
| Gewicht gleich, n = 30, Quote > 50 % | 0,467 | 2,2 % (3/4) | 0,0 % (4/4) | 0,0 % (4/4) | 1,0 % (0/4) | 0,0 % (4/4) |
| Gewicht gestuft, n = 30, Quote > 60 % | 0,395 | 17,8 % (0/4) | 0,5 % (3/4) | 2,4 % (0/4) | 17,1 % (0/4) | 0,0 % (4/4) |
| Gewicht Pareto-Tail, n = 30, Quote > 50 % | 0,499 | 10,4 % (0/4) | 1,0 % (0/4) | 3,6 % (0/4) | 9,8 % (0/4) | 0,0 % (4/4) |
| Gewicht Pareto-Tail, n = 100, Quote > 50 % | 0,499 | 27,0 % (0/4) | 2,9 % (0/4) | 18,5 % (0/4) | 17,9 % (0/4) | 0,0 % (4/4) |
| Gewicht gleichverteilt 1 bis 100, n = 30, Quote > 30 % | 0,700 | 16,0 % (0/4) | 5,9 % (0/4) | 2,3 % (0/4) | 9,0 % (0/4) | 0,0 % (4/4) |
| Gewicht gleichverteilt 1 bis 100, n = 30, Quote > 50 % | 0,500 | 8,2 % (0/4) | 0,8 % (0/4) | 2,5 % (0/4) | 5,4 % (0/4) | 0,0 % (4/4) |
| Gewicht gleichverteilt 1 bis 100, n = 30, Quote > 70 % | 0,300 | 68,4 % (0/4) | 9,7 % (0/4) | 2,9 % (0/4) | 19,1 % (0/4) | 0,0 % (4/4) |
| Standort zufällig, n = 20, m = 6, Fixkosten 60 | 0,000 | 1,8 % (0/4) | 1,1 % (0/4) | 1,5 % (0/4) | 3,0 % (0/4) | 0,3 % (3/4) |
| Dreiecks-Typen mit Zufall, n = 30, Fixkosten 30 | 0,071 | 10,6 % (0/4) | 5,2 % (0/4) | 2,8 % (0/4) | 4,6 % (0/4) | 0,3 % (3/4) |

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen des Einzelplans waren Hypothesen aus einer Vormessung mit anderem Zufallsgenerator; hier mit SplitMix64 neu gemessen. Es gilt die Messung.

- **H1 bis H3, H6 und H10 (im Gewichtsspiel) bestätigt, mit anderen Zahlen:** Zeilenerzeugung gleich volles LP (Plan: 138 Vergleiche, hier im Scratch-Lauf 1 306), Gewichtsspiel etwa linear (Plan: 2,2 bis 3,3 Zeilen je Spieler bei n = 100; hier 2,1 bis 3,5 ab n = 20), Veto-Regel (Plan: 810 von 810; hier nur an 48 Instanzen im Repo und 276 im Scratch-Lauf neu geprüft, nicht an 810).
- **V8-Abbruchzahlen bleiben widerlegt:** die Obergrenze verschweigt die Zahl (E2). Hier n = 20: 2 von 15 abgebrochen, 229,9 Zeilen ohne Obergrenze (Plan: 312,1, größter Wert 1 002; andere Instanzen); n = 30: 6 von 10 abgebrochen, 790,8 (Plan: 786,3, 7 von 10 abgebrochen). Der Befund hält in beiden Messungen: bei n = 30 brach mehr als die Hälfte der Läufe ab, und das Mittel ohne Obergrenze war mehr als doppelt so groß wie mit; die einzelnen Zahlen halten nicht.
- **Standortspiel bei n = 40:** Plan 3 568 ± 1 413 Zeilen (k = 5), hier 5749 ± 3127 bei n = 40 (Mittel ± Std über 4 Seeds): andere Instanzen und wenige Seeds, die Streuung ist groß.
- **H8 (mehr Zufallskoalitionen verbessern die Lücke):** gemessen sind nur die zwei Stufen 500 und 5 000; im Gewichtsspiel ja, im Standortspiel gilt die Ausnahme des Plans (Dreiecks-Typen: 500 → 5 000 Koalitionen verbesserte nicht) hier **nicht**: in den Dreiecks-Typen mit Zufall verbessern 5 000 statt 500 Koalitionen die Lücke, siehe Tabelle; ob die Verbesserung stetig verläuft, ist nicht gemessen. Die Ausnahme gilt also nicht für diese Instanzen.
- **Standort mit 12 Standorten und leerem Kern (Plan: 10 von 120 Instanzen bei n = 20, Fixkosten 200):** hier nicht bestätigt, 0 von 50 Instanzen (Scratch-Lauf, Seeds 910 bis 959). Die dafür vorgesehene Zelle der E3-Tabelle ist gestrichen; die Spalten des Standortspiels stützen sich auf die Zelle mit 6 Standorten und die Dreiecks-Typen mit Zufall.
- **H9 (Stichproben-Schätzung verlässlich):** widerlegt, wie im Plan (E4).
- **H11 (Core Lagrangian nicht besser als Stichproben-LP):** hier gemessen am Nachbau, nicht am Verfahren der Autoren; in 8 von 9 Zellen der Tabelle ist die mittlere Lücke des Nachbaus größer als die des Stichproben-LP mit 5 000 Koalitionen. Beim Veto-Preset erreicht er die Ecke x = (1, 0, …) nicht (die App weist das aus).
- **H12 (Nukleolus bei großem n):** nicht gebaut (Empfehlung des Plans), nur Verweis auf `kern-demo`.
- **Vehikel mit leerem Kern bei n ≥ 20:** wie im Plan nur das Gewichtsspiel und die Dreiecks-Typen; das zufällige Standortspiel hatte in den gemessenen Einstellungen selten einen leeren Kern (Tabelle oben).
- **Neu gegenüber dem Plan:** Eigene Gewichtsliste, Erzeuger „gleichverteilt“ (1 bis 100), Live-Vergleich auf dem Spiel der Seitenleiste, Zeile-für-Zeile-Ansicht, Teilungsprüfung für ε\* ≥ 0. Ein Fehler der ersten Fassung, den der Randfalltest fand: die Rucksack-Trennung ist unzulässig, wenn nur die große Koalition gewinnt; sie meldet dann keine Verletzung statt abzustürzen.

## Ehrliche Grenzen

**Annahmen**

- Genossenschaft (Gewichtsspiel): ganzzahlige Stimmgewichte, ein Beschluss braucht **mehr als** den eingestellten Prozentsatz der Gewichtssumme. v(S) = 1 für beschlussfähige Gruppen, sonst 0; Aufteilungen sind nichtnegativ und verteilen den Wert 1. Die Gewichte kommen aus Erzeugern (gleich, gestuft, Pareto-Tail, gleichverteilt) oder aus der eigenen Liste.
- Standortspiel: jeder Kunde ist ein Spieler, c(S) sind die kleinsten Kosten aus Fixkosten der offenen Standorte plus Wegen der Kunden von S zu ihrem nächsten offenen Standort; ohne Kapazitäten, gleiche Fixkosten je Standort, Lage zufällig im Quadrat oder als Dreiecks-Typen (konstruiert).
- ε\* ist der Least-Core-Wert: das kleinste ε, für das eine Aufteilung des Gesamtwerts jede echte nichtleere Koalition mit Überschuss höchstens ε hält. Kern leer ⇔ ε\* > 0, Kern am Rand ⇔ ε\* = 0, Kern mit Innerem ⇔ ε\* < 0; Schwelle für „gleich null“: 10⁻⁶ von Wert(N), mindestens 10⁻⁶ absolut.
- Die Zeilenerzeugung hört auf, wenn keine Koalition ε um mehr als 10⁻⁷ von Wert(N) (mindestens 10⁻⁷) übertrifft. Die Trennung ist im Gewichtsspiel ein Rucksack-MILP (HiGHS), im Standortspiel die Aufzählung aller Standortmengen; der MILP läuft mit der voreingestellten relativen Lücke 1e-4 von HiGHS, die Aufzählung prüft alle Standortmengen. Wegen der Lücke kann die gelieferte Koalition etwas unter dem Optimum des Rucksack-Problems liegen: in einem Scratch-Lauf mit 300 Aufrufen (n = 12) einmal, um 2,7e-5. Mit Lücke 0 wären in den Messreihen (Gewichtsspiele) einzelne Zeilenzahlen um höchstens 3 Zeilen anders ausgefallen (Pareto-Tail und gleichverteilte Gewichte, n = 30 bis 100), die Lücke einer Stichproben-LP-Instanz um 1,9e-5 von Wert(N); die Aussagen blieben gleich.
- Aller Zufall (Instanzen, Stichproben, Heuristik) kommt aus SplitMix64 mit dem eingestellten Zufalls-Seed.

**Grenzen**

- Berechnet wird der Least-Core-Wert und eine Aufteilung dazu, nicht der Nukleolus (dafür `kern-demo` bis n = 10) und nicht der Shapley-Wert des Standortspiels.
- Der Core-Lagrangian-Nachbau ist ein Nachbau nach einer Beschreibung des Verfahrens, die Hyperparameter sind abgestimmt bzw. geraten, die Algorithmen und der Code der Autoren sind nicht nachgebaut. Aus seinem Abschneiden folgt nichts über das Verfahren der Autoren.
- Die gezielte Heuristik ist je Spielklasse eigens gebaut (Rucksack-Greedy, Einzelflip-Suche) und beweist nichts; nur die Zeilenerzeugung mit Trennung über alle Koalitionen liefert ε\* (bis auf die Abbruchschwelle).
- Zufällige Standortspiele hatten bei n = 10 bis 20 in den gemessenen Einstellungen nur selten (0 bis 1 von 40 bis 60) einen leeren Kern; der leere Kern kommt hier aus dem Gewichtsspiel und aus den konstruierten Dreiecks-Typen.
- Die Aussagen gelten für die gemessenen Erzeuger, Quoten und Raster, nicht für jedes Spiel.

**Bekannte Probleme und Unsicherheiten**

- Welche optimale Ecke HiGHS liefert, hängt von Plattform und Bibliotheksstand ab: Aufteilung, Zeilenzahl und Zahl der Schleifen sind dadurch Bänder; ε\* selbst nicht.
- Zeiten sind Messungen auf einem Rechner unter wechselnder Last und nur als Größenordnung zu lesen; Live-Läufe zeigen ihre Zeit am Ergebnis.
- Die Messreihen haben je Zelle wenige Instanzen (meist vier Zufalls-Seeds); Mittel und Streuung sind grob.
- Die Live-Läufe brechen nach 5 000 Schleifen ab und weisen das aus; die vorgerechneten Messreihen haben keine Obergrenze, außer wo sie ausdrücklich als Gegenprobe genannt ist.
- Die Shapley-Shubik-Zähltabelle ist nur bis zu einer Tabellengröße von drei Millionen Einträgen (n mal Gewichtssumme) rechenbar; darüber meldet die App, dass sie nicht rechnet.
- Literatur: Titel, Autoren, Jahr und Fundstelle sind über Verzeichnisse bestätigt; Volltexte wurden nicht gelesen, bei Gemp u. a. nur das Abstract.

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur der Least-Core-Wert, nicht der Nukleolus** | Der Nukleolus braucht eine LP-Folge; die Demo rechnet ihn nicht. Wie die Zeilenerzeugung dort hilft, ist hier nicht gemessen. | `kern-demo` rechnet ihn bis n = 10 |
| **Core-Lagrangian ist ein Nachbau** | Das Abschneiden des Nachbaus sagt nichts über das Verfahren der Autoren; Algorithmen und Hyperparameter der Arbeit sind nicht übernommen. | – |
| **Ein Kunde je Spieler, ohne Kapazitäten, gleiche Fixkosten** | Mit Kapazitäten oder ungleichen Fixkosten ändern sich c(S) und der Kern; die Messreihen gelten nicht dafür. | `standortspiel-demo` (Linie Koalitionen und Teilen) |
| **Einfaches Spiel mit Gewinn 0 oder 1** | Bei allgemeinen Koalitionswerten (etwa Flüsse) ist die Trennung ein anderes Problem als das Rucksack-Problem; die Zeilenerzeugung bleibt dieselbe. | `flussspiel-demo` (Linie Koalitionen und Teilen) |
| **Aufteilung nach Shapley-Shubik statt Least Core** | Der Index ist keine Kern-Aufteilung; die Brücke zeigt nur, wie weit er vom Least Core liegt. | `shapley-demo` für den Shapley-Wert, `machtindex-demo` (Linie Koalitionen und Teilen) für Machtindizes |
| **Zufall im Standortspiel: leerer Kern selten** | Für den leeren Kern braucht es die konstruierten Dreiecks-Typen oder hohe Fixkosten mit vielen Standorten. | – |

**Literatur** (Titel, Autoren, Jahr und Fundstelle über Verzeichnisse bestätigt; Volltexte nicht gelesen, bei Gemp u. a. nur das Abstract)

- Gemp, I., Lanctot, M., Marris, L., Mao, Y., Duéñez-Guzmán, E., Perrin, S., Gyorgy, A., Elie, R., Piliouras, G., Kaisers, M., Hennes, D., Bullard, K., Larson, K., Bachrach, Y. (2024): Approximating the Core via Iterative Coalition Sampling. AAMAS 2024, arXiv 2402.03928 (Abstract gelesen).
- Maschler, M., Peleg, B., Shapley, L. S. (1979): Geometric Properties of the Kernel, Nucleolus, and Related Solution Concepts. Mathematics of Operations Research 4(4), 303–338 (Fundstelle bestätigt, Inhalt nicht gelesen).
- Shapley, L. S., Shubik, M. (1954): A Method for Evaluating the Distribution of Power in a Committee System. American Political Science Review 48(3), 787–792.
- Zeilenerzeugung (englisch row generation) für Koalitionsbedingungen: Literatur nicht geprüft.

## Tests

178 Tests (`pytest --collect-only -q tests`), Laufzeit einmal gemessen rund 66 Sekunden mit `-n 6` (Windows; bei paralleler Last dauert es länger). Zusammensetzung: Mini-Instanzen von Hand (`test_algorithm.py`), Orakel gegen volles LP, Rucksack-DP, Standort-MILP, geschlossene Formeln, Ganzzahligkeitslücke und Shapley-Shubik-Aufzählung (`test_oracle_kern_zeilenerzeugung.py`, kurze Fassung; der Scratch-Lauf mit 1 306 Vergleichen steht nur in dieser README), Randfälle n = 1, n = 2, kein Gewinner außer N (`test_edge_cases.py`), Verdikte je Schwelle und Zahl (`test_verdicts.py`), Zahl-zu-Test-Tabelle der README (`test_claims.py`), Messreihen: Hash, Aufbau, Neu-Rechnen als Band (`test_messreihen.py`), Presets mit Verdikt-Klasse und Permalink (`test_presets.py`), Layout und Hervorhebung (`test_layout.py`), AppTest mit jedem Knopf, jedem Regler an Minimum und Maximum, Permalink und Worst-Case n = 100 (`test_app.py`), Wortgleichheit App und README (`test_texts.py`), Kopietest der SplitMix64-Quelle (`test_copies.py`), Footer und Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | Repo- und Streamlit-Einstellungen |
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, Zeile für Zeile, Experimente, Brücke, Grenzen, Formeln |
| `kzl_games.py` | Spielklassen `GewichtsSpiel` und `StandortSpiel` (Koalitionswert, Überschuss, Veto-Regel, offene Standorte) |
| `kzl_lp.py` | Least-Core-LP über eine Zeilenmenge, volles LP als Orakel |
| `kzl_separation.py` | Trennungen: Rucksack-MILP, Aufzählung der Standortmengen, Standort-MILP |
| `kzl_rowgen.py` | Zeilenerzeugung mit Protokoll je Schleife |
| `kzl_sampling.py` | Stichproben-LP, Core-Lagrangian-Nachbau, gezielte Heuristik, wahre und geschätzte Verletzung |
| `kzl_power.py` | Shapley-Shubik per Zähl-DP |
| `kzl_scenario.py`, `kzl_stream.py`, `kzl_rng.py` | Instanzen, Zufallsströme, SplitMix64 (wörtliche Kopie) |
| `kzl_messreihen.py`, `kzl_auswertung.py`, `data/messreihen.json`, `tools/precompute_messreihen.py` | Raster E1 bis E5, Auswertung, vorgerechnete Tabelle mit Erzeuger-Hash |
| `kzl_evaluation.py` | Einstellungen, Analyse, Formatierung, alle Verdikte als reine Funktionen |
| `kzl_constants.py`, `kzl_presets.py`, `kzl_layout.py`, `kzl_texts.py`, `kzl_visualization.py` | Grenzen und Presets, Permalink, fester Kopf, feste Texte, Plotly-Figuren |
| `tests/` | Mini-Instanzen, Orakel, Randfälle, Verdikte, Claims, Messreihen, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Kopietest |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (Streamlit, NumPy, Plotly, SciPy), plus pytest |

## Bewusst nicht umgesetzt

- **Nukleolus bei großem n:** die Zeilenerzeugung spart dort nach der Vormessung des Plans bei n bis 12 nichts; `kern-demo` rechnet ihn bis n = 10.
- **Algorithmen 1 und 2 der Autoren (Projektion, SGD) und ihr Code:** nicht nachgebaut, nur der Nachbau aus einer Beschreibung.
- **Kapazitäten, ungleiche Fixkosten, mehrere Kunden je Spieler** im Standortspiel; allgemeine Koalitionswerte (Flüsse): `standortspiel-demo` und `flussspiel-demo` der Linie Koalitionen und Teilen.
- **PDF-Export:** die Demo hat keinen.
- **Aussagen über reale Gremien, Anlage- oder Rechtsberatung:** Stimmgewichte, Quoten und Koalitionswerte sind Modellannahmen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Laufzeit: Python 3.12 in der CI, `scipy>=1.9` (wegen `milp`); die in der Streamlit Cloud eingestellte Python-Version ist beim Deploy zu prüfen.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Koalitionen und Teilen: Macht und Aufteilung](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html).
