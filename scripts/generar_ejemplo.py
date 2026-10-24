"""Genera data/ventas_ejemplo.csv: datos sintéticos con faltantes, atípicos y duplicados a propósito."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

sedes = rng.choice(["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga"], n, p=[.35, .25, .2, .12, .08])
productos = rng.choice(["Café", "Arepa", "Jugo", "Empanada", "Pandebono", "Té"], n)
precio_base = {"Café": 4500, "Arepa": 3000, "Jugo": 5500, "Empanada": 2500, "Pandebono": 2000, "Té": 4000}

cantidad = rng.poisson(6, n) + 1
precio = np.array([precio_base[p] for p in productos]) * rng.normal(1, 0.08, n)
descuento = np.clip(rng.normal(5, 3, n), 0, None).round(1)
total = cantidad * precio * (1 - descuento / 100)
satisfaccion = np.clip(rng.normal(4.1, 0.6, n) - descuento * 0.02, 1, 5).round(1)
fechas = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")

df = pd.DataFrame({
    "fecha": fechas.strftime("%Y-%m-%d"),
    "sede": sedes,
    "producto": productos,
    "cantidad": cantidad,
    "precio_unitario": precio.round(0),
    "descuento_pct": descuento,
    "total": total.round(0),
    "satisfaccion": satisfaccion,
    "canal": rng.choice(["Tienda", "Domicilio", "App"], n, p=[.6, .25, .15]),
})

# Atípicos: pedidos corporativos enormes y un error de captura (precio x100).
big = rng.choice(n, 8, replace=False)
df.loc[big, "cantidad"] = rng.integers(80, 150, 8)
df.loc[big, "total"] = (df.loc[big, "cantidad"] * df.loc[big, "precio_unitario"]).round(0)
df.loc[17, "precio_unitario"] *= 100

# Faltantes con distintas "formas" de vacío.
df = df.astype(object)
for col, frac, token in [("satisfaccion", .18, ""), ("descuento_pct", .07, "NA"), ("canal", .05, "sin dato"),
                         ("sede", .02, "-")]:
    idx = rng.choice(n, int(n * frac), replace=False)
    df.loc[idx, col] = token

df = pd.concat([df, df.iloc[[3, 50, 120]]], ignore_index=True)  # duplicados

out = Path(__file__).resolve().parents[1] / "data" / "ventas_ejemplo.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"{len(df)} filas -> {out}")
