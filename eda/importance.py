"""Importancia de las variables.

Sin variable objetivo, "importancia" significa cuánta información aporta una
columna: qué tan completa está, cuánto varía y qué tanto se relaciona con las
demás. Con objetivo, se mide la asociación de cada columna con él.

Asociaciones entre tipos mixtos, todas en [0, 1]:
- numérica–numérica: |Spearman| (robusta a atípicos y relaciones no lineales monótonas)
- categórica–numérica: razón de correlación η
- categórica–categórica: V de Cramér
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ROWS = 20000
MAX_CATEGORIES = 20


def _role(s: pd.Series) -> str:
    n = s.notna().sum()
    nunique = s.nunique(dropna=True)
    if n == 0:
        return "vacía"
    if nunique <= 1:
        return "constante"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "fecha"
    if nunique == 2:
        return "binaria"
    if pd.api.types.is_numeric_dtype(s):
        is_int = np.allclose(s.dropna() % 1, 0)
        if is_int and n >= 20 and nunique / n > 0.95 and s.dropna().is_monotonic_increasing:
            return "identificador"
        return "numérica discreta" if is_int and nunique <= 15 else "numérica continua"
    if n >= 20 and nunique / n > 0.95:
        mean_len = s.dropna().astype(str).str.len().mean()
        return "texto libre" if mean_len > 30 else "identificador"
    return "categórica"


USABLE_ROLES = {"fecha", "binaria", "numérica discreta", "numérica continua", "categórica"}


def _is_categorical(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))


def _as_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s - s.min()).dt.total_seconds() / 86400
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return s.astype(float)


def _as_category(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    top = s.value_counts().index[:MAX_CATEGORIES]
    return s.where(s.isin(top) | s.isna(), "Otros")


def _entropy(counts: np.ndarray) -> float:
    """Entropía de Shannon normalizada a [0, 1]."""
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def variability(s: pd.Series) -> float:
    s = s.dropna()
    if s.nunique() <= 1:
        return 0.0
    if _is_categorical(s):
        return _entropy(_as_category(s).value_counts().to_numpy())
    x = _as_numeric(s)
    # Recorta al percentil 1-99: un solo atípico extremo aplastaría el histograma.
    lo, hi = x.quantile([0.01, 0.99])
    if hi > lo:
        x = x.clip(lo, hi)
    bins = min(20, x.nunique())
    return _entropy(np.histogram(x, bins=bins)[0])


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(_as_category(a), _as_category(b))
    if min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / n
    chi2 = ((observed - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    groups = _as_category(cat)
    y = _as_numeric(num)
    total = ((y - y.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    between = y.groupby(groups).agg(lambda g: len(g) * (g.mean() - y.mean()) ** 2).sum()
    return float(np.sqrt(between / total))


def association(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) < 5 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan, ""
    cat_a, cat_b = _is_categorical(a), _is_categorical(b)
    if cat_a and cat_b:
        return cramers_v(a, b), "V de Cramér"
    if cat_a:
        return correlation_ratio(a, b), "η"
    if cat_b:
        return correlation_ratio(b, a), "η"
    # Spearman = Pearson sobre los rangos (evita depender de scipy).
    r = _as_numeric(a).rank().corr(_as_numeric(b).rank())
    return abs(float(r)), "|Spearman|"


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(MAX_ROWS, random_state=0) if len(df) > MAX_ROWS else df


def association_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = _sample(df)
    m = pd.DataFrame(np.nan, index=columns, columns=columns)
    for i, a in enumerate(columns):
        m.loc[a, a] = 1.0
        for b in columns[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = association(df[a], df[b])[0]
    return m


def variable_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna: rol, calidad, variabilidad y puntaje de información (0-100)."""
    roles = {c: _role(df[c]) for c in df.columns}
    usable = [c for c in df.columns if roles[c] in USABLE_ROLES]
    assoc = association_matrix(df, usable) if len(usable) > 1 else pd.DataFrame()

    rows = []
    for col in df.columns:
        s = df[col]
        completeness = float(s.notna().mean()) if len(s) else 0.0
        var = variability(s) if roles[col] in USABLE_ROLES else 0.0
        if not assoc.empty and col in assoc.index:
            others = assoc.loc[col].drop(col).dropna()
            best = (others.idxmax(), float(others.max())) if not others.empty else (None, 0.0)
        else:
            best = (None, 0.0)
        score = 100 * completeness * var * (0.6 + 0.4 * best[1]) if roles[col] in USABLE_ROLES else 0.0
        rows.append({
            "columna": col,
            "rol": roles[col],
            "completitud_pct": round(100 * completeness, 1),
            "únicos": int(s.nunique(dropna=True)),
            "variabilidad": round(var, 3),
            "más_relacionada_con": best[0],
            "asociación_máx": round(best[1], 3),
            "puntaje_info": round(score, 1),
        })
    return pd.DataFrame(rows).set_index("columna").sort_values("puntaje_info", ascending=False)


def target_importance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Asociación de cada columna utilizable con la variable objetivo, de mayor a menor."""
    df = _sample(df)
    rows = []
    for col in df.columns:
        if col == target or _role(df[col]) not in USABLE_ROLES:
            continue
        value, measure = association(df[col], df[target])
        if pd.notna(value):
            rows.append({"columna": col, "asociación": round(value, 3), "medida": measure})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("columna").sort_values("asociación", ascending=False)


def examples(df: pd.DataFrame, n: int = 3) -> dict[str, list]:
    """Valores representativos por columna, para que el LLM infiera qué representa cada una."""
    out = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            out[col] = []
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[col] = [s.min().strftime("%Y-%m-%d"), s.max().strftime("%Y-%m-%d")]
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = [round(float(v), 4) for v in (s.min(), s.median(), s.max())]
        else:
            out[col] = [str(v)[:40] for v in s.astype(str).value_counts().index[:n]]
    return out
