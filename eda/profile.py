"""Reúne todo el análisis de un DataFrame en un solo objeto."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import importance, quality, stats


@dataclass
class Profile:
    overview: dict
    types: dict[str, list[str]]
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    datetime: pd.DataFrame
    missing: pd.DataFrame
    missing_overview: dict
    outliers: pd.DataFrame
    outlier_method: str
    outlier_k: float
    correlation: pd.DataFrame
    strong_pairs: list[dict]
    alerts: list[str]
    variables: pd.DataFrame
    examples: dict[str, list]
    target: str | None = None
    target_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cleaning_actions: list[str] = field(default_factory=list)
    dataset_name: str = "dataset"

    def to_dict(self, max_columns: int = 40) -> dict:
        """Versión compacta y serializable, pensada para el prompt del LLM."""

        def frame(df: pd.DataFrame) -> dict:
            if df.empty:
                return {}
            return json.loads(df.head(max_columns).to_json(orient="index", date_format="iso", double_precision=4))

        missing = self.missing[self.missing["faltantes"] > 0]
        return {
            "nombre": self.dataset_name,
            "resumen": self.overview,
            "columnas_por_tipo": {k: v[:max_columns] for k, v in self.types.items() if v},
            "limpieza": self.cleaning_actions,
            "variables": frame(self.variables),
            "variable_objetivo": self.target,
            "importancia_respecto_al_objetivo": frame(self.target_importance),
            "numericas": frame(self.numeric),
            "categoricas": frame(self.categorical),
            "fechas": frame(self.datetime),
            "faltantes": {
                **self.missing_overview,
                "por_columna": frame(missing[["faltantes", "pct"]]) if not missing.empty else {},
            },
            "atipicos": {
                "metodo": self.outlier_method,
                "k": self.outlier_k,
                "por_columna": frame(self.outliers[self.outliers["atípicos"] > 0]) if not self.outliers.empty else {},
            },
            "correlaciones_fuertes": self.strong_pairs[:15],
            "alertas": self.alerts,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(**kwargs), ensure_ascii=False, default=_json_default, indent=2)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def profile_dataframe(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    outlier_k: float | None = None,
    corr_threshold: float = 0.5,
    cleaning_actions: list[str] | None = None,
    dataset_name: str = "dataset",
    target: str | None = None,
) -> Profile:
    k = quality.default_k(outlier_method) if outlier_k is None else outlier_k
    if target is not None and target not in df.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe en los datos.")
    corr, pairs = stats.correlations(df, corr_threshold)
    return Profile(
        overview=stats.overview(df),
        types=stats.column_types(df),
        numeric=stats.numeric_summary(df),
        categorical=stats.categorical_summary(df),
        datetime=stats.datetime_summary(df),
        missing=quality.missing_with_suggestions(df),
        missing_overview=quality.missing_overview(df),
        outliers=quality.detect_outliers(df, outlier_method, k),
        outlier_method=outlier_method,
        outlier_k=k,
        correlation=corr,
        strong_pairs=pairs,
        alerts=quality.quality_alerts(df),
        variables=importance.variable_overview(df),
        examples=importance.examples(df),
        target=target,
        target_importance=importance.target_importance(df, target) if target else pd.DataFrame(),
        cleaning_actions=cleaning_actions or [],
        dataset_name=dataset_name,
    )
