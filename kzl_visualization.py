"""Plotly-Figuren (alle Achsen fest, `fixedrange`; Beschriftungen als Annotationen)."""

import numpy as np
import plotly.graph_objects as go

import kzl_constants as C


def _fest(fig: go.Figure, hoehe: int = 320) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=hoehe, margin=dict(l=50, r=20, t=30, b=50), legend=dict(orientation="h", y=-0.28))
    return fig


def build_eps_curve(zeilen_vorher: list[int], eps: list[float], eps_stern: float) -> go.Figure:
    """ε des LP gegen die Zahl der Zeilen: steigt monoton und endet bei ε*."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zeilen_vorher, y=eps, mode="lines+markers", line=dict(color=C.COLOR_LINE), marker=dict(size=4), name="ε des LP"))
    fig.add_hline(y=eps_stern, line=dict(color=C.COLOR_ALT, dash="dash"))
    fig.add_annotation(x=zeilen_vorher[-1], y=eps_stern, text="ε*", showarrow=False, yshift=12, font=dict(color=C.COLOR_ALT), bgcolor="rgba(255,255,255,0.7)")
    fig.update_xaxes(title="Zeilen im LP")
    fig.update_yaxes(title="ε")
    return _fest(fig)


def build_allocation(x, gewicht_anteil=None, einzelkosten=None, kind: str = "gewinn") -> go.Figure:
    """Aufteilung je Spieler (Balken); Gewichtsspiel: dazu der Stimmanteil w_i / W, Kostenspiel: die Einzelkosten c({i})."""
    idx = np.arange(1, len(x) + 1)
    fig = go.Figure(go.Bar(x=idx, y=x, marker_color=C.COLOR_CHOSEN, name="Aufteilung x"))
    if gewicht_anteil is not None:
        fig.add_trace(go.Scatter(x=idx, y=gewicht_anteil, mode="markers", marker=dict(color=C.COLOR_ALT, size=6), name="Stimmanteil"))
    if einzelkosten is not None:
        fig.add_trace(go.Scatter(x=idx, y=einzelkosten, mode="markers", marker=dict(color=C.COLOR_ALT, size=6), name="Kosten allein"))
    fig.update_xaxes(title="Spieler", dtick=max(1, len(x) // 10))
    fig.update_yaxes(title="Anteil am Wert" if kind == "gewinn" else "Kostenanteil")
    return _fest(fig)


def build_rows_vs_n(reihen: dict[str, dict[int, float]], logy: bool = False) -> go.Figure:
    """Zeilen gegen n, eine Linie je Erzeuger bzw. Vehikel."""
    fig = go.Figure()
    for name, reihe in reihen.items():
        ns = sorted(reihe)
        fig.add_trace(go.Scatter(x=ns, y=[reihe[n] for n in ns], mode="lines+markers", name=name))
    fig.update_xaxes(title="Spieler n")
    fig.update_yaxes(title="Zeilen im LP (Mittel über die Seeds)", type="log" if logy else "linear")
    return _fest(fig)


def build_gap_bars(zellen: list[str], werte: dict[str, list[float]]) -> go.Figure:
    """Gruppierte Balken: Lücke gegen ε* (Anteil von Wert(N)) je Zelle und Verfahren."""
    fig = go.Figure()
    for verfahren, ys in werte.items():
        fig.add_trace(go.Bar(x=zellen, y=ys, name=C.E3_LABEL.get(verfahren, verfahren)))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Lücke gegen ε* (Anteil von Wert(N))")
    fig.update_xaxes(tickangle=-30)
    return _fest(fig, 380)


def build_regime(punkte: dict[str, list[tuple[int, float]]]) -> go.Figure:
    """Zeilen gegen n je Regime (ein offener / mehrere offene Standorte)."""
    fig = go.Figure()
    for name, pts in punkte.items():
        pts = sorted(pts)
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=name))
    fig.update_xaxes(title="Kunden n")
    fig.update_yaxes(title="Zeilen im LP (Mittel)")
    return _fest(fig)


def build_bruecke(x_lc, phi) -> go.Figure:
    """Least-Core-Aufteilung gegen Shapley-Shubik je Spieler."""
    idx = np.arange(1, len(phi) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=x_lc, name="Least Core (Zeilenerzeugung)", marker_color=C.COLOR_CHOSEN))
    fig.add_trace(go.Bar(x=idx, y=phi, name="Shapley-Shubik", marker_color=C.COLOR_ALT))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Spieler", dtick=max(1, len(phi) // 10))
    fig.update_yaxes(title="Anteil am Wert")
    return _fest(fig)
