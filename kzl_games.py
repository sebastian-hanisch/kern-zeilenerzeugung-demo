"""Die zwei Spielklassen der Demo, mit gleicher Schnittstelle (DEMO-PLAYBOOK §1b: eine Einheit je Regel).

GewichtsSpiel  (Gewinnspiel, `kind = "gewinn"`): n Spieler, ganzzahlige Gewichte w, ganzzahlige Quote q; v(S) = 1, wenn w(S) >= q, sonst 0.
                Aufteilung x >= 0 mit x(N) = v(N) = 1; Überschuss einer Koalition e(S, x) = v(S) - x(S).
StandortSpiel  (Kostenspiel, `kind = "kosten"`): n Kunden, m mögliche Standorte, Entfernungen d (n x m), Fixkosten fix (m);
                c(S) = min über nichtleere Standortmengen F von fix(F) + Summe über i in S von min_{j in F} d_ij.
                Aufteilung x mit x(N) = c(N); Überschuss e(S, x) = x(S) - c(S).
In beiden gilt: ε* = kleinstes ε, für das eine Aufteilung alle echten nichtleeren Koalitionen mit Überschuss höchstens ε hält (Least Core).
Der Kern ist leer genau dann, wenn ε* > 0; er liegt am Rand, wenn ε* = 0 (nicht leer, aber ohne Inneres); er hat ein Inneres, wenn ε* < 0.
`n_eval` zählt die ausgewerteten Koalitionswerte (Einheit: eine Koalition)."""

import numpy as np

CHUNK_ELEMENTE = 4_000_000   # höchstens so viele Matrixelemente je Zwischenergebnis (Speichergrenze bei großen Stapeln)


def schluessel(mask) -> bytes:
    """Hashbarer Schlüssel einer Koalition (Bool-Maske)."""
    return np.packbits(np.asarray(mask, dtype=np.uint8)).tobytes()


def alle_koalitionen(n: int) -> np.ndarray:
    """Alle 2^n - 2 echten nichtleeren Koalitionen als Bool-Matrix (nur für kleines n: Orakel und volle LP)."""
    if n > 20:
        raise ValueError("alle_koalitionen: n > 20 würde mehr als eine Million Zeilen erzeugen")
    ks = np.arange(1, (1 << n) - 1)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


class GewichtsSpiel:
    kind = "gewinn"

    def __init__(self, w, q):
        self.w = np.asarray(w, dtype=float)
        self.n = len(self.w)
        self.q = float(q)
        self.n_eval = 0
        self.vN = 1.0 if self.w.sum() >= self.q else 0.0

    def value_batch(self, S, count: bool = True) -> np.ndarray:
        S = np.asarray(S, dtype=bool)
        if count:
            self.n_eval += len(S)
        return (S.astype(float) @ self.w >= self.q - 0.5).astype(float)

    def excess(self, x, S) -> np.ndarray:
        return self.value_batch(S) - np.asarray(S, dtype=float) @ np.asarray(x, dtype=float)

    def has_veto_player(self) -> bool:
        """Handregel: Spieler i ist Veto-Spieler, wenn die übrigen zusammen die Quote nicht erreichen (w(N) - w_i < q)."""
        return bool(np.any(self.w.sum() - self.w < self.q - 0.5))


class StandortSpiel:
    kind = "kosten"

    def __init__(self, d, fix):
        self.d = np.asarray(d, dtype=float)
        self.n, self.m = self.d.shape
        self.fix = np.asarray(fix, dtype=float)
        ks = np.arange(1, 1 << self.m)
        self.fm = ((ks[:, None] >> np.arange(self.m)[None, :]) & 1).astype(bool)      # (nF, m): Standortmengen
        self.D = np.where(self.fm[:, None, :], self.d[None, :, :], np.inf).min(axis=2)  # (nF, n): kleinster Weg je Kunde in der Menge
        self.fk = self.fm @ self.fix                                                  # (nF,): Fixkosten der Menge
        self.n_eval = 0
        self.vN = float(self._values(np.ones((1, self.n), dtype=bool))[0])

    def _values(self, S) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        out = np.empty(len(S))
        schritt = max(1, CHUNK_ELEMENTE // len(self.fk))
        for a in range(0, len(S), schritt):
            out[a:a + schritt] = (self.fk[None, :] + S[a:a + schritt] @ self.D.T).min(axis=1)
        return out

    def value_batch(self, S, count: bool = True) -> np.ndarray:
        S = np.asarray(S, dtype=bool)
        if count:
            self.n_eval += len(S)
        return self._values(S)

    def excess(self, x, S) -> np.ndarray:
        return np.asarray(S, dtype=float) @ np.asarray(x, dtype=float) - self.value_batch(S)

    def open_facilities(self) -> np.ndarray:
        """Indizes der offenen Standorte im Optimum der großen Koalition (bei Gleichstand die Menge mit der kleinsten Nummer)."""
        f = int(np.argmin(self.fk + self.D.sum(axis=1)))
        return np.flatnonzero(self.fm[f])

    def cost_of_facility_set(self, S, f_index: int) -> float:
        """Kosten der Koalition S, wenn genau die Standortmenge mit Nummer f_index (Zeile von fm) benutzt wird."""
        return float(self.fk[f_index] + np.asarray(S, dtype=float) @ self.D[f_index])
