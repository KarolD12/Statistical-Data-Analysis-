"""Estadísticas descriptivas por tipo de columna."""

from __future__ import annotations

import numpy as np
import pandas as pd


def column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    types: dict[str, list[str]] = {"numérica": [], "categórica": [], "fecha": [], "booleana": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            types["booleana"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            types["numérica"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            types["fecha"].append(col)
        else:
            types["categórica"].append(col)
    return types


def overview(df: pd.DataFrame) -> dict:
    types = column_types(df)
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "filas_duplicadas": int(df.duplicated().sum()),
        "memoria_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "tipos": {k: len(v) for k, v in types.items() if v},
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    desc = pd.DataFrame({
        "conteo": num.count(),
        "media": num.mean(),
        "desv_est": num.std(),
        "mín": num.min(),
        "q1": num.quantile(0.25),
        "mediana": num.median(),
        "q3": num.quantile(0.75),
        "máx": num.max(),
        "asimetría": num.skew(),
        "curtosis": num.kurt(),
    })
    mean = desc["media"].replace(0, np.nan)
    desc["coef_var"] = (desc["desv_est"] / mean.abs())
    return desc.round(4)


def categorical_summary(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols if cols is not None else column_types(df)["categórica"] + column_types(df)["booleana"]
    rows = []
    for col in cols:
        s = df[col].dropna().astype(str)
        if s.empty:
            rows.append({"columna": col, "conteo": 0, "únicos": 0, "más_frecuente": None,
                         "frecuencia": 0, "pct_más_frecuente": 0.0})
            continue
        vc = s.value_counts()
        rows.append({
            "columna": col,
            "conteo": int(s.size),
            "únicos": int(vc.size),
            "más_frecuente": vc.index[0],
            "frecuencia": int(vc.iloc[0]),
            "pct_más_frecuente": round(100 * vc.iloc[0] / s.size, 2),
        })
    return pd.DataFrame(rows).set_index("columna") if rows else pd.DataFrame()


def datetime_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = column_types(df)["fecha"]
    rows = []
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append({
            "columna": col,
            "conteo": int(s.size),
            "mín": s.min(),
            "máx": s.max(),
            "rango_días": int((s.max() - s.min()).days),
        })
    return pd.DataFrame(rows).set_index("columna") if rows else pd.DataFrame()


def correlations(df: pd.DataFrame, threshold: float = 0.5) -> tuple[pd.DataFrame, list[dict]]:
    """Matriz de Pearson y los pares con |r| >= threshold, de mayor a menor."""
    num = df.select_dtypes(include="number")
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return pd.DataFrame(), []
    corr = num.corr()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"var_1": a, "var_2": b, "r": round(float(r), 3)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return corr.round(3), pairs
