"""Calidad de datos: valores faltantes, atípicos y alertas."""

from __future__ import annotations

import pandas as pd

OUTLIER_METHODS = ("iqr", "zscore", "mad")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    counts = df.isna().sum()
    out = pd.DataFrame({
        "faltantes": counts,
        "pct": (100 * counts / n).round(2) if n else 0.0,
    })
    return out.sort_values("faltantes", ascending=False)


def missing_overview(df: pd.DataFrame) -> dict:
    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    return {
        "celdas_faltantes": missing_cells,
        "pct_celdas": round(100 * missing_cells / total_cells, 2) if total_cells else 0.0,
        "filas_con_faltantes": int(df.isna().any(axis=1).sum()),
        "columnas_con_faltantes": int(df.isna().any().sum()),
    }


def _bounds(s: pd.Series, method: str, k: float) -> tuple[float, float]:
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr
    if method == "zscore":
        mu, sd = s.mean(), s.std()
        return mu - k * sd, mu + k * sd
    if method == "mad":
        # z modificado (Iglewicz y Hoaglin): 0.6745 * (x - mediana) / MAD
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return med, med
        delta = k * mad / 0.6745
        return med - delta, med + delta
    raise ValueError(f"Método desconocido: {method}. Usa uno de {OUTLIER_METHODS}.")


def default_k(method: str) -> float:
    return {"iqr": 1.5, "zscore": 3.0, "mad": 3.5}[method]


def detect_outliers(
    df: pd.DataFrame, method: str = "iqr", k: float | None = None, min_unique: int = 5
) -> pd.DataFrame:
    """Un renglón por columna numérica con límites y número de atípicos.

    Se omiten columnas con menos de `min_unique` valores distintos (binarias,
    códigos), donde "atípico" no tiene sentido.
    """
    k = default_k(method) if k is None else k
    rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if s.nunique() < min_unique:
            continue
        low, high = _bounds(s, method, k)
        if low == high:
            continue
        mask = (s < low) | (s > high)
        out = s[mask]
        rows.append({
            "columna": col,
            "atípicos": int(mask.sum()),
            "pct": round(100 * mask.mean(), 2),
            "límite_inf": round(float(low), 4),
            "límite_sup": round(float(high), 4),
            "bajo": int((s < low).sum()),
            "alto": int((s > high).sum()),
            "ejemplos": sorted(out.round(4).unique().tolist(), key=lambda v: -abs(v - s.median()))[:5],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("columna").sort_values("atípicos", ascending=False)


def outlier_mask(df: pd.DataFrame, col: str, method: str = "iqr", k: float | None = None) -> pd.Series:
    k = default_k(method) if k is None else k
    s = df[col]
    low, high = _bounds(s.dropna(), method, k)
    return ((s < low) | (s > high)).fillna(False)


def quality_alerts(df: pd.DataFrame, missing_threshold: float = 50.0) -> list[str]:
    alerts = []
    n = len(df)
    if n == 0:
        return ["El conjunto de datos no tiene filas."]
    dups = int(df.duplicated().sum())
    if dups:
        alerts.append(f"Hay {dups} filas duplicadas ({100 * dups / n:.1f}%).")
    for col in df.columns:
        s = df[col]
        pct_na = 100 * s.isna().mean()
        if pct_na == 100:
            continue
        if pct_na >= missing_threshold:
            alerts.append(f"'{col}' tiene {pct_na:.1f}% de valores faltantes.")
        nunique = s.nunique(dropna=True)
        if nunique == 1:
            alerts.append(f"'{col}' es constante (un solo valor); no aporta información.")
        elif (
            not pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_datetime64_any_dtype(s)
            and n >= 20
            and nunique / s.notna().sum() > 0.95
        ):
            alerts.append(f"'{col}' tiene casi un valor distinto por fila; parece un identificador o texto libre.")
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            skew = s.skew()
            if pd.notna(skew) and abs(skew) > 2:
                alerts.append(f"'{col}' tiene una distribución muy asimétrica (asimetría {skew:.2f}).")
            if (s < 0).any() and (s >= 0).mean() > 0.95:
                alerts.append(f"'{col}' es casi siempre no negativa pero tiene {int((s < 0).sum())} valores negativos.")
    return alerts


def fill_suggestion(s: pd.Series) -> str:
    """Sugerencia sencilla de tratamiento para faltantes de una columna."""
    pct = 100 * s.isna().mean()
    if pct == 0:
        return "—"
    if pct > 60:
        return "Considerar eliminar la columna"
    if pd.api.types.is_numeric_dtype(s):
        skew = s.skew()
        return "Imputar con la mediana" if pd.notna(skew) and abs(skew) > 1 else "Imputar con la media"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "Interpolar o eliminar filas"
    return "Imputar con la moda o categoría 'Desconocido'"


def missing_with_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    out = missing_summary(df)
    out["sugerencia"] = [fill_suggestion(df[c]) for c in out.index]
    out["tipo"] = [str(df[c].dtype) for c in out.index]
    return out[["tipo", "faltantes", "pct", "sugerencia"]]
