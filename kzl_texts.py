"""Feste Texte der App, die auch in der README stehen (ein Test vergleicht sie): Annahmen, Grenzen, bekannte Probleme, Erklärtabelle, Literatur.

Zahlen stehen hier nur, wenn sie Einstellungen sind (Regler, Raster) oder von Hand herleitbar; Messzahlen kommen mit f-String aus der Rechnung oder stehen in der README mit Testverweis."""

EINLEITUNG = """\
Teil der [Linie Koalitionen und Teilen](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html). Eine Gruppe von Spielern teilt einen Wert, und der **Kern** verlangt, dass **keine** der 2ⁿ − 2 möglichen Gruppen austreten will, weil sie allein besser führe.
`kern-demo` schreibt diese Bedingungen bis n = 10 alle auf; bei 30 Spielern wären es über eine Milliarde. Hier geht es bis n = 100: Ein **LP mit wenigen Bedingungen** wird gelöst, dann wird **gezielt** die am stärksten verletzte Koalition gesucht und als neue Bedingung nachgelegt (**Zeilenerzeugung**). Zum Vergleich ziehen andere Verfahren Koalitionen **zufällig**.

Zwei Vehikel: eine **Genossenschaft** mit Stimmgewichten (Gewinnspiel; bei den gemessenen Erzeugern und Quoten meist leerer Kern) und ein **Standortspiel** (Kunden teilen Fixkosten, leerer Kern in den Messungen selten). Alle Stimmgewichte, Quoten und Koalitionswerte sind **Modellannahmen**, keine Aussage über reale Gremien. Weiter unten: die Experimente **🔬**, **🌉 Brücke**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""

ERKLAERTABELLE = """\
| Begriff / Verfahren | Was es ist | Beispielspur (Gewichte 4, 3, 2, 1, Quote 6) |
|---|---|---|
| **Kern** | Aufteilungen des Werts, bei denen keine Gruppe mit ihrem Austritt mehr als ihren Anteil bekäme. | Gewinner sind {1,2}, {1,3}, {2,3,4} und alle Obermengen; jede Aufteilung verletzt mindestens eine davon: der Kern ist leer. |
| **Least Core, ε\\*** | Der kleinste Zuschlag ε, um den jede Gruppe ihren Austrittswert höchstens übertreffen darf. Kern leer ⇔ ε\\* > 0. Der Wert ε\\* heißt hier auch Preis der Stabilität. | ε\\* = 1/3 bei x = (1/3, 1/3, 1/3, 0): die drei Bedingungen für {1,2}, {1,3}, {2,3,4} summieren sich zu 3ε ≥ 1. |
| **Zeilenerzeugung** | LP mit wenigen Zeilen lösen, die am stärksten verletzte Koalition suchen, als Zeile nachlegen, wiederholen, bis keine mehr verletzt ist. Dann gilt ε\\* des vollen LP. | Start mit den 4 Einer-Zeilen (ε = −1/4), in 3 Schleifen kommen 3 Zeilen dazu, die vierte Schleife findet keine Verletzung. |
| **Trennungsproblem** | Die Suche nach der am stärksten verletzten Koalition. Im Gewichtsspiel ein Rucksack-Problem (kleinstes x(S) mit w(S) ≥ q), im Standortspiel eine Aufzählung der Standortmengen oder ein MILP. | Für x = (0,4; 0,3; 0,2; 0,1): kleinste Summe 0,6 bei {1,3} oder {2,3,4}, Überschuss 0,4. |
| **Stichproben-LP** | Nur K zufällig gezogene Koalitionen als Zeilen. Das LP ist schnell, hält aber nur die gezogenen Bedingungen. | Mit wenigen gezogenen Koalitionen kann {2,3,4} fehlen: dann liegt ε unter ε\\*. |
| **Core-Lagrangian-Nachbau** | Eigener Nachbau eines Verfahrens aus der Literatur (Stapel zufälliger Koalitionen, Mirror Prox auf einer Lagrange-Funktion); Nachbau nach einer Beschreibung, nicht nach dem Code der Autoren. | – |
| **Gezielte Heuristik** | LP plus Trennung durch eine Heuristik (Rucksack-Greedy bzw. Einzelflip-Suche): findet meist verletzte Koalitionen, beweist aber nichts. | – |
| **Shapley-Shubik-Index** | Anteil der Reihenfolgen, in denen ein Mitglied die Gruppe erstmals beschlussfähig macht. | 5/12, 3/12, 3/12, 1/12. |
"""

ASSUMPTIONS_MARKDOWN = """\
**Annahmen**

- Genossenschaft (Gewichtsspiel): ganzzahlige Stimmgewichte, ein Beschluss braucht **mehr als** den eingestellten Prozentsatz der Gewichtssumme. v(S) = 1 für beschlussfähige Gruppen, sonst 0; Aufteilungen sind nichtnegativ und verteilen den Wert 1. Die Gewichte kommen aus Erzeugern (gleich, gestuft, Pareto-Tail, gleichverteilt) oder aus der eigenen Liste.
- Standortspiel: jeder Kunde ist ein Spieler, c(S) sind die kleinsten Kosten aus Fixkosten der offenen Standorte plus Wegen der Kunden von S zu ihrem nächsten offenen Standort; ohne Kapazitäten, gleiche Fixkosten je Standort, Lage zufällig im Quadrat oder als Dreiecks-Typen (konstruiert).
- ε\\* ist der Least-Core-Wert: das kleinste ε, für das eine Aufteilung des Gesamtwerts jede echte nichtleere Koalition mit Überschuss höchstens ε hält. Kern leer ⇔ ε\\* > 0, Kern am Rand ⇔ ε\\* = 0, Kern mit Innerem ⇔ ε\\* < 0; Schwelle für „gleich null“: 10⁻⁶ von Wert(N), mindestens 10⁻⁶ absolut.
- Die Zeilenerzeugung hört auf, wenn keine Koalition ε um mehr als 10⁻⁷ von Wert(N) (mindestens 10⁻⁷) übertrifft. Die Trennung ist im Gewichtsspiel ein Rucksack-MILP (HiGHS), im Standortspiel die Aufzählung aller Standortmengen; der MILP läuft mit der voreingestellten relativen Lücke 1e-4 von HiGHS, die Aufzählung prüft alle Standortmengen. Wegen der Lücke kann die gelieferte Koalition etwas unter dem Optimum des Rucksack-Problems liegen: in einem Scratch-Lauf mit 300 Aufrufen (n = 12) einmal, um 2,7e-5. Mit Lücke 0 wären in den Messreihen (Gewichtsspiele) einzelne Zeilenzahlen um höchstens 3 Zeilen anders ausgefallen (Pareto-Tail und gleichverteilte Gewichte, n = 30 bis 100), die Lücke einer Stichproben-LP-Instanz um 1,9e-5 von Wert(N); die Aussagen blieben gleich.
- Aller Zufall (Instanzen, Stichproben, Heuristik) kommt aus SplitMix64 mit dem eingestellten Zufalls-Seed.

**Grenzen**

- Berechnet wird der Least-Core-Wert und eine Aufteilung dazu, nicht der Nukleolus (dafür `kern-demo` bis n = 10) und nicht der Shapley-Wert des Standortspiels.
- Der Core-Lagrangian-Nachbau ist ein Nachbau nach einer Beschreibung des Verfahrens, die Hyperparameter sind abgestimmt bzw. geraten, die Algorithmen und der Code der Autoren sind nicht nachgebaut. Aus seinem Abschneiden folgt nichts über das Verfahren der Autoren.
- Die gezielte Heuristik ist je Spielklasse eigens gebaut (Rucksack-Greedy, Einzelflip-Suche) und beweist nichts; nur die Zeilenerzeugung mit Trennung über alle Koalitionen liefert ε\\* (bis auf die Abbruchschwelle).
- Zufällige Standortspiele hatten bei n = 10 bis 20 in den gemessenen Einstellungen nur selten (0 bis 1 von 40 bis 60) einen leeren Kern; der leere Kern kommt hier aus dem Gewichtsspiel und aus den konstruierten Dreiecks-Typen.
- Die Aussagen gelten für die gemessenen Erzeuger, Quoten und Raster, nicht für jedes Spiel.

**Bekannte Probleme und Unsicherheiten**

- Welche optimale Ecke HiGHS liefert, hängt von Plattform und Bibliotheksstand ab: Aufteilung, Zeilenzahl und Zahl der Schleifen sind dadurch Bänder; ε\\* selbst nicht.
- Zeiten sind Messungen auf einem Rechner unter wechselnder Last und nur als Größenordnung zu lesen; Live-Läufe zeigen ihre Zeit am Ergebnis.
- Die Messreihen haben je Zelle wenige Instanzen (meist vier Zufalls-Seeds); Mittel und Streuung sind grob.
- Die Live-Läufe brechen nach 5 000 Schleifen ab und weisen das aus; die vorgerechneten Messreihen haben keine Obergrenze, außer wo sie ausdrücklich als Gegenprobe genannt ist.
- Die Shapley-Shubik-Zähltabelle ist nur bis zu einer Tabellengröße von drei Millionen Einträgen (n mal Gewichtssumme) rechenbar; darüber meldet die App, dass sie nicht rechnet.
- Literatur: Titel, Autoren, Jahr und Fundstelle sind über Verzeichnisse bestätigt; Volltexte wurden nicht gelesen, bei Gemp u. a. nur das Abstract.
"""

GRENZEN_TABELLE = """\
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur der Least-Core-Wert, nicht der Nukleolus** | Der Nukleolus braucht eine LP-Folge; die Demo rechnet ihn nicht. Wie die Zeilenerzeugung dort hilft, ist hier nicht gemessen. | `kern-demo` rechnet ihn bis n = 10 |
| **Core-Lagrangian ist ein Nachbau** | Das Abschneiden des Nachbaus sagt nichts über das Verfahren der Autoren; Algorithmen und Hyperparameter der Arbeit sind nicht übernommen. | – |
| **Ein Kunde je Spieler, ohne Kapazitäten, gleiche Fixkosten** | Mit Kapazitäten oder ungleichen Fixkosten ändern sich c(S) und der Kern; die Messreihen gelten nicht dafür. | `standortspiel-demo` (Linie Koalitionen und Teilen) |
| **Einfaches Spiel mit Gewinn 0 oder 1** | Bei allgemeinen Koalitionswerten (etwa Flüsse) ist die Trennung ein anderes Problem als das Rucksack-Problem; die Zeilenerzeugung bleibt dieselbe. | `flussspiel-demo` (Linie Koalitionen und Teilen) |
| **Aufteilung nach Shapley-Shubik statt Least Core** | Der Index ist keine Kern-Aufteilung; die Brücke zeigt nur, wie weit er vom Least Core liegt. | `shapley-demo` für den Shapley-Wert, `machtindex-demo` (Linie Koalitionen und Teilen) für Machtindizes |
| **Zufall im Standortspiel: leerer Kern selten** | Für den leeren Kern braucht es die konstruierten Dreiecks-Typen oder hohe Fixkosten mit vielen Standorten. | – |
"""

LITERATUR_HEAD = "**Literatur** (Titel, Autoren, Jahr und Fundstelle über Verzeichnisse bestätigt; Volltexte nicht gelesen, bei Gemp u. a. nur das Abstract)"
LITERATUR = [
    "- Gemp, I., Lanctot, M., Marris, L., Mao, Y., Duéñez-Guzmán, E., Perrin, S., Gyorgy, A., Elie, R., Piliouras, G., Kaisers, M., Hennes, D., Bullard, K., Larson, K., Bachrach, Y. (2024): Approximating the Core via Iterative Coalition Sampling. AAMAS 2024, arXiv 2402.03928 (Abstract gelesen).",
    "- Maschler, M., Peleg, B., Shapley, L. S. (1979): Geometric Properties of the Kernel, Nucleolus, and Related Solution Concepts. Mathematics of Operations Research 4(4), 303–338 (Fundstelle bestätigt, Inhalt nicht gelesen).",
    "- Shapley, L. S., Shubik, M. (1954): A Method for Evaluating the Distribution of Power in a Committee System. American Political Science Review 48(3), 787–792.",
    "- Zeilenerzeugung (englisch row generation) für Koalitionsbedingungen: Literatur nicht geprüft.",
]
