"""Gráficos interactivos con Plotly.

Colores por función: un solo azul para magnitudes, rojo reservado para
atípicos (siempre con etiqueta en la leyenda), divergente azul–gris–rojo para
correlaciones.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import quality

SERIES = "#2a78d6"
SERIES_LIGHT = "#86b6ef"
CRITICAL = "#d03b3b"
SURFACE = "#fcfcfb"
PRESENT = "#f0efec"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, title: str, height: int = 380, **kwargs) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": TEXT}},
        font={"family": FONT, "size": 12, "color": TEXT_2},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        margin={"l": 56, "r": 24, "t": 56, "b": 48},
        hoverlabel={"bgcolor": "white", "font": {"family": FONT, "color": TEXT}, "bordercolor": GRID},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont={"color": MUTED})
    return fig


def histogram(df: pd.DataFrame, col: str, bins: int = 30) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df[col].dropna(), nbinsx=bins,
        marker={"color": SERIES, "line": {"color": SURFACE, "width": 2}},
        hovertemplate=f"{col}: %{{x}}<br>Frecuencia: %{{y}}<extra></extra>",
    ))
    median = df[col].median()
    fig.add_vline(x=median, line={"color": TEXT_2, "width": 1, "dash": "dot"},
                  annotation_text=f"mediana {median:,.2f}", annotation_font_color=TEXT_2)
    fig.update_layout(bargap=0.04, barcornerradius=4)
    return _layout(fig, f"Distribución de {col}", xaxis_title=col, yaxis_title="Frecuencia")


def box_with_outliers(df: pd.DataFrame, col: str, method: str = "iqr", k: float | None = None) -> go.Figure:
    s = df[col]
    mask = quality.outlier_mask(df, col, method, k)
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=s.dropna(), name=col, boxpoints=False, orientation="h",
        marker_color=SERIES, line={"width": 2}, fillcolor="rgba(42,120,214,0.15)",
        hoverinfo="skip", showlegend=False,
    ))
    normal, outl = s[~mask].dropna(), s[mask]
    fig.add_trace(go.Scatter(
        x=normal, y=[col] * len(normal), mode="markers", name="Normal",
        marker={"color": SERIES_LIGHT, "size": 8, "opacity": 0.5, "line": {"color": SURFACE, "width": 2}},
        customdata=normal.index, hovertemplate="Fila %{customdata}<br>%{x}<extra>Normal</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=outl, y=[col] * len(outl), mode="markers", name=f"Atípico ({len(outl)})",
        marker={"color": CRITICAL, "size": 10, "symbol": "diamond", "line": {"color": SURFACE, "width": 2}},
        customdata=outl.index, hovertemplate="Fila %{customdata}<br>%{x}<extra>Atípico</extra>",
    ))
    fig.update_yaxes(showticklabels=False)
    return _layout(fig, f"Atípicos en {col} (método {method.upper()})", height=260, xaxis_title=col)


def missing_bar(df: pd.DataFrame) -> go.Figure | None:
    miss = quality.missing_summary(df)
    miss = miss[miss["faltantes"] > 0].sort_values("pct")
    if miss.empty:
        return None
    fig = go.Figure(go.Bar(
        x=miss["pct"], y=miss.index.astype(str), orientation="h",
        marker={"color": SERIES, "line": {"color": SURFACE, "width": 2}},
        text=[f"{p:.1f}%" for p in miss["pct"]], textposition="outside",
        textfont={"color": TEXT_2},
        customdata=miss["faltantes"],
        hovertemplate="%{y}<br>%{customdata} faltantes (%{x:.1f}%)<extra></extra>",
    ))
    fig.update_layout(barcornerradius=4, bargap=0.35)
    fig.update_xaxes(range=[0, min(100, miss["pct"].max() * 1.2 + 5)], ticksuffix="%")
    return _layout(fig, "Porcentaje de valores faltantes por columna",
                   height=max(260, 32 * len(miss) + 120), xaxis_title="% de filas")


def missing_matrix(df: pd.DataFrame, max_rows: int = 400) -> go.Figure | None:
    if not df.isna().any().any():
        return None
    view = df if len(df) <= max_rows else df.sample(max_rows, random_state=0).sort_index()
    z = view.isna().astype(int).T.values
    fig = go.Figure(go.Heatmap(
        z=z, x=view.index, y=[str(c) for c in view.columns],
        colorscale=[[0, PRESENT], [1, SERIES]], zmin=0, zmax=1, showscale=False,
        xgap=0, ygap=2,
        customdata=[["Faltante" if v else "Presente" for v in row] for row in z],
        hovertemplate="Fila %{x}<br>%{y}: %{customdata}<extra></extra>",
    ))
    title = "Mapa de faltantes (azul = faltante)"
    if len(df) > max_rows:
        title += f" — muestra de {max_rows} filas"
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return _layout(fig, title, height=max(260, 24 * df.shape[1] + 120), xaxis_title="Fila")


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure | None:
    if corr.empty:
        return None
    n = len(corr)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale=DIVERGING, zmin=-1, zmax=1, xgap=2, ygap=2,
        text=corr.round(2).values if n <= 12 else None,
        texttemplate="%{text}" if n <= 12 else None,
        textfont={"color": TEXT},
        colorbar={"title": "r", "thickness": 12},
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    size = max(380, 40 * n + 160)
    return _layout(fig, "Correlación de Pearson", height=size)


def score_bar(values: pd.Series, title: str, xaxis_title: str, max_value: float | None = None) -> go.Figure:
    """Barras horizontales ordenadas de mayor (arriba) a menor."""
    values = values.sort_values()
    fig = go.Figure(go.Bar(
        x=values.values, y=values.index.astype(str), orientation="h",
        marker={"color": SERIES, "line": {"color": SURFACE, "width": 2}},
        text=[f"{v:.2f}" if max_value == 1 else f"{v:.0f}" for v in values.values],
        textposition="outside", textfont={"color": TEXT_2},
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(barcornerradius=4, bargap=0.35)
    if max_value is not None:
        fig.update_xaxes(range=[0, max_value * 1.12])
    return _layout(fig, title, height=max(260, 30 * len(values) + 120), xaxis_title=xaxis_title)


def category_bar(df: pd.DataFrame, col: str, top: int = 10) -> go.Figure:
    vc = df[col].dropna().astype(str).value_counts()
    if len(vc) > top:
        vc = pd.concat([vc.head(top), pd.Series({"Otros": vc.iloc[top:].sum()})])
    vc = vc.iloc[::-1]
    pct = 100 * vc / vc.sum()
    fig = go.Figure(go.Bar(
        x=vc.values, y=vc.index, orientation="h",
        marker={"color": [MUTED if i == "Otros" else SERIES for i in vc.index],
                "line": {"color": SURFACE, "width": 2}},
        customdata=pct, hovertemplate="%{y}<br>%{x} filas (%{customdata:.1f}%)<extra></extra>",
    ))
    fig.update_layout(barcornerradius=4, bargap=0.35)
    return _layout(fig, f"Valores más frecuentes de {col}",
                   height=max(260, 30 * len(vc) + 120), xaxis_title="Filas")
