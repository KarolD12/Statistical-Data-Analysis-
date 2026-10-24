"""Carga y limpieza robusta de archivos CSV.

Un CSV no declara tipos, ni qué significa un valor vacío, ni qué separador o
codificación usa. Este módulo detecta esas cosas y deja un registro de cada
decisión tomada, para que el informe pueda explicarlas.
"""

from __future__ import annotations

import csv
import io
import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

# Cadenas que, en la práctica, significan "no hay dato".
NA_TOKENS = [
    "", " ", "NA", "N/A", "n/a", "na", "NaN", "nan", "NAN", "null", "NULL", "Null",
    "None", "none", "-", "--", "?", "#N/A", "#N/D", "N/D", "n/d", "s/d", "S/D",
    "sin dato", "Sin dato", "SIN DATO", "missing",
]

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
DELIMITERS = [",", ";", "\t", "|"]

# Proporción mínima de valores convertibles para cambiar el tipo de una columna.
CONVERSION_THRESHOLD = 0.9


@dataclass
class LoadResult:
    df: pd.DataFrame
    encoding: str
    separator: str
    decimal: str
    actions: list[str] = field(default_factory=list)


def detect_encoding(raw: bytes) -> str:
    for enc in ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_separator(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        # Si el sniffer falla, gana el delimitador más frecuente en la primera línea.
        first_line = sample.splitlines()[0] if sample else ""
        return max(DELIMITERS, key=first_line.count)


def detect_decimal(sample: str, separator: str) -> str:
    """Si el separador no es coma y aparecen números como 12,5 la coma es decimal."""
    if separator != "," and re.search(r"(?<![\w,])\d+,\d+(?![\w,])", sample):
        return ","
    return "."


def _to_numeric(series: pd.Series, decimal: str) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace(r"[\$€£%\s]", "", regex=True)
    if decimal == ",":
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _looks_like_date(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    pattern = r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
    return sample.str.match(pattern).mean() >= CONVERSION_THRESHOLD


def _to_datetime(series: pd.Series) -> pd.Series:
    sample = series.dropna().astype(str)
    # Año primero (ISO) no debe interpretarse con dayfirst.
    dayfirst = not sample.str.match(r"^\d{4}").mean() > 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def clean_dataframe(df: pd.DataFrame, decimal: str = ".") -> tuple[pd.DataFrame, list[str]]:
    """Normaliza nombres, quita filas/columnas vacías e infiere tipos."""
    actions: list[str] = []
    df = df.copy()

    original = list(df.columns)
    df.columns = [str(c).strip() or f"columna_{i + 1}" for i, c in enumerate(df.columns)]
    renamed = sum(a != b for a, b in zip(original, df.columns))
    if renamed:
        actions.append(f"Se normalizaron {renamed} nombres de columna (espacios o vacíos).")

    unnamed = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    to_drop = sorted(set(unnamed) | set(empty_cols), key=list(df.columns).index)
    if to_drop:
        df = df.drop(columns=to_drop)
        actions.append(f"Se eliminaron {len(to_drop)} columnas completamente vacías: {', '.join(to_drop)}.")

    n_before = len(df)
    df = df.dropna(how="all")
    if len(df) < n_before:
        actions.append(f"Se eliminaron {n_before - len(df)} filas completamente vacías.")
    df = df.reset_index(drop=True)

    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        stripped = s.astype("string").str.strip()
        stripped = stripped.mask(stripped.isin(NA_TOKENS))
        non_null = stripped.notna().sum()
        if non_null == 0:
            df[col] = stripped.astype(object)
            continue

        numeric = _to_numeric(stripped, decimal)
        ratio = numeric.notna().sum() / non_null
        if ratio >= CONVERSION_THRESHOLD:
            lost = int(non_null - numeric.notna().sum())
            df[col] = numeric
            msg = f"'{col}' se convirtió a numérica"
            if lost:
                msg += f" ({lost} valores no numéricos pasaron a faltantes)"
            actions.append(msg + ".")
            continue

        if _looks_like_date(stripped):
            dates = _to_datetime(stripped)
            if dates.notna().sum() / non_null >= CONVERSION_THRESHOLD:
                df[col] = dates
                actions.append(f"'{col}' se convirtió a fecha.")
                continue

        df[col] = stripped.astype(object).where(stripped.notna(), None)

    return df, actions


def load_csv(source, max_bytes: int | None = None) -> LoadResult:
    """Lee un CSV desde una ruta, bytes o un objeto tipo archivo."""
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        with open(source, "rb") as fh:
            raw = fh.read()

    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError(f"El archivo supera el tamaño máximo permitido ({max_bytes / 1e6:.0f} MB).")
    if not raw.strip():
        raise ValueError("El archivo está vacío.")

    encoding = detect_encoding(raw)
    text = raw.decode(encoding)
    sample = text[:20000]
    separator = detect_separator(sample)
    decimal = detect_decimal(sample, separator)

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            dtype=str,
            keep_default_na=False,
            na_values=NA_TOKENS,
            skipinitialspace=True,
            on_bad_lines="warn",
            engine="python",
        )
    except pd.errors.ParserError as exc:
        raise ValueError(f"No se pudo interpretar el CSV: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise ValueError("El CSV no contiene filas de datos.")

    df, actions = clean_dataframe(df, decimal)
    sep_name = {",": "coma", ";": "punto y coma", "\t": "tabulador", "|": "barra vertical"}[separator]
    actions.insert(0, f"Codificación detectada: {encoding}; separador: {sep_name}; decimal: '{decimal}'.")
    return LoadResult(df=df, encoding=encoding, separator=separator, decimal=decimal, actions=actions)
