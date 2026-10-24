"""Uso por línea de comandos.

    python -m eda datos.csv -o informe.md --html informe.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import llm, plots
from .loader import load_csv
from .profile import profile_dataframe
from .quality import OUTLIER_METHODS
from .report import generate_report


def build_html(report_md: str, figures) -> str:
    import markdown

    body = markdown.markdown(report_md, extensions=["tables"])
    figures = [f for f in figures if f is not None]
    # Solo el primer gráfico carga plotly.js (desde la CDN, en la versión que usa la librería).
    charts = "\n".join(
        f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figures)
    )
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe EDA</title>
<style>body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:960px;margin:2rem auto;
padding:0 16px;color:#0b0b0b;background:#fcfcfb;line-height:1.55}}h1,h2,h3{{line-height:1.2}}</style>
</head><body>{body}<h2>Gráficos</h2>{charts}</body></html>"""


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m eda", description="Análisis exploratorio de un CSV.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Ruta del informe Markdown (por defecto, salida estándar).")
    parser.add_argument("--html", type=Path, help="Además, guarda un informe HTML con gráficos.")
    parser.add_argument("--method", choices=OUTLIER_METHODS, default="iqr", help="Método de atípicos.")
    parser.add_argument("--k", type=float, default=None, help="Umbral del método de atípicos.")
    parser.add_argument("--provider", choices=llm.PROVIDERS, default=None,
                        help="groq (nube) u ollama (local). Por defecto, LLM_PROVIDER o groq.")
    parser.add_argument("--model", default=None, help="Modelo del proveedor (p. ej. llama3.2:3b en Ollama).")
    parser.add_argument("--target", default=None, help="Variable objetivo para medir la importancia.")
    parser.add_argument("--context", default="", help="Descripción breve de los datos para el LLM.")
    parser.add_argument("--no-llm", action="store_true", help="No usar LLM; informe basado en reglas.")
    args = parser.parse_args(argv)

    try:
        loaded = load_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        profile = profile_dataframe(
            loaded.df, args.method, args.k, cleaning_actions=loaded.actions, dataset_name=args.csv.name,
            target=args.target,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    report, source = generate_report(
        profile, use_llm=not args.no_llm, context=args.context, provider=args.provider, model=args.model
    )
    print(f"Informe generado ({source}).", file=sys.stderr)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Guardado en {args.output}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(report)

    if args.html:
        df = loaded.df
        figures = [
            plots.score_bar(profile.variables["puntaje_info"], "Puntaje de información (0-100)", "Puntaje", 100),
            plots.missing_bar(df),
            plots.correlation_heatmap(profile.correlation),
        ]
        figures += [plots.box_with_outliers(df, c, args.method, args.k) for c in profile.outliers.index[:6]]
        args.html.write_text(build_html(report, figures), encoding="utf-8")
        print(f"HTML guardado en {args.html}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
