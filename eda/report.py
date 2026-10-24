"""Redacción del informe en lenguaje natural.

El informe lo escribe un LLM (Groq en la nube u Ollama en local) a partir del
perfil estadístico. Si no hay ninguno disponible, o la llamada falla, se genera
un informe basado en reglas, para que la herramienta siga siendo útil sin conexión.
"""

from __future__ import annotations

import json

from . import llm
from .profile import Profile

# Se conservan por compatibilidad con código que los importe desde aquí.
DEFAULT_MODEL = llm.GROQ_DEFAULT
AVAILABLE_MODELS = llm.GROQ_MODELS

SYSTEM_PROMPT = """Eres un analista de datos senior. Recibes el perfil estadístico de un \
conjunto de datos en formato JSON y redactas un informe de análisis exploratorio en español, \
claro y profesional, para una persona que no necesariamente es experta en estadística.

Reglas:
- Usa solo los números del JSON. No inventes valores, columnas ni conclusiones que los datos no respalden.
- Redondea con sentido (1-2 decimales) y explica qué significa cada hallazgo, no solo el número.
- Correlación no implica causalidad: dilo si mencionas relaciones entre variables.
- Si algo no se puede concluir con la información disponible, dilo.
- Formato Markdown con estas secciones, en este orden:
  ## Resumen general
  ## Variables y su importancia
  ## Calidad de los datos
  ### Valores faltantes
  ### Valores atípicos
  ## Hallazgos principales
  ## Relaciones entre variables
  ## Recomendaciones
- En "Variables y su importancia" usa el puntaje de información y, si existe, la importancia \
respecto a la variable objetivo.
- Las recomendaciones deben ser concretas (qué columna, qué tratamiento, por qué)."""

VARIABLES_PROMPT = """Eres un analista de datos senior. Recibes, en JSON, la lista de columnas \
de un conjunto de datos con su rol, completitud, variabilidad, puntaje de información (0-100), la \
variable con la que más se relaciona y algunos valores de ejemplo.

Escribe en español una tabla Markdown con las columnas:
| Variable | Qué representa | Importancia para el análisis |

- "Qué representa": infiérelo del nombre y los ejemplos, en una frase. Si no es claro, dilo.
- "Importancia para el análisis": alta, media o baja, y por qué en una frase (usa el puntaje, \
la completitud y sus relaciones; un identificador o una constante tiene importancia baja para el análisis).
- Una fila por variable, en el mismo orden del JSON. No inventes columnas.
Después de la tabla, un párrafo de 2-3 frases sobre qué variables conviene mirar primero y por qué.
Responde solo con la tabla y el párrafo."""


def build_prompt(profile: Profile, context: str = "") -> str:
    parts = []
    if context.strip():
        parts.append(f"Contexto proporcionado por el usuario sobre los datos:\n{context.strip()}\n")
    parts.append("Perfil estadístico del conjunto de datos:\n```json\n" + profile.to_json() + "\n```")
    parts.append("Redacta el informe.")
    return "\n".join(parts)


def _variables_payload(profile: Profile) -> str:
    rows = json.loads(profile.variables.to_json(orient="index", force_ascii=False))
    for col, row in rows.items():
        row["ejemplos"] = profile.examples.get(col, [])
    payload = {"variables": rows}
    if profile.target:
        payload["variable_objetivo"] = profile.target
        payload["importancia_respecto_al_objetivo"] = json.loads(profile.target_importance.to_json(orient="index"))
    return json.dumps(payload, ensure_ascii=False)


def generate_llm_report(profile: Profile, context: str = "", temperature: float = 0.3, **llm_kwargs) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(profile, context)},
    ]
    return llm.chat(messages, temperature=temperature, **llm_kwargs)


def describe_variables_llm(profile: Profile, context: str = "", **llm_kwargs) -> str:
    user = ""
    if context.strip():
        user += f"Contexto de los datos: {context.strip()}\n\n"
    user += _variables_payload(profile)
    messages = [{"role": "system", "content": VARIABLES_PROMPT}, {"role": "user", "content": user}]
    return llm.chat(messages, temperature=0.2, max_tokens=3000, **llm_kwargs)


ROLE_TEXT = {
    "identificador": "Identificador de registro; sirve para unir tablas, no para analizar.",
    "constante": "Tiene un solo valor; no aporta información.",
    "vacía": "No tiene datos.",
    "texto libre": "Texto libre; requiere procesamiento de lenguaje para analizarse.",
    "fecha": "Variable temporal; permite estudiar tendencias y estacionalidad.",
    "binaria": "Variable de dos valores; útil para segmentar o como objetivo de clasificación.",
    "numérica discreta": "Conteo o escala numérica con pocos valores.",
    "numérica continua": "Medida numérica continua.",
    "categórica": "Variable categórica; útil para agrupar y comparar.",
}


def _level(score: float) -> str:
    return "alta" if score >= 60 else "media" if score >= 30 else "baja"


def describe_variables_rules(profile: Profile) -> str:
    lines = ["| Variable | Rol | Importancia | Comentario |", "|---|---|---|---|"]
    for col, r in profile.variables.iterrows():
        comment = ROLE_TEXT.get(r["rol"], "")
        if r["completitud_pct"] < 100:
            comment += f" Completa al {r['completitud_pct']}%."
        if r["más_relacionada_con"] and r["asociación_máx"] >= 0.5:
            comment += f" Muy relacionada con {r['más_relacionada_con']} ({r['asociación_máx']})."
        lines.append(f"| **{col}** | {r['rol']} | {_level(r['puntaje_info'])} ({r['puntaje_info']}) | {comment.strip()} |")
    top = [c for c, r in profile.variables.iterrows() if r["puntaje_info"] >= 60][:3]
    if top:
        lines += ["", f"Las variables con más información son: {', '.join(f'**{c}**' for c in top)}."]
    return "\n".join(lines)


def describe_variables(profile: Profile, use_llm: bool = True, context: str = "", **llm_kwargs) -> tuple[str, str]:
    """Devuelve (descripción, origen). El origen es 'llm' o 'reglas'."""
    provider = llm_kwargs.get("provider") or llm.default_provider()
    if use_llm and llm.is_available(provider, llm_kwargs.get("api_key")):
        try:
            return describe_variables_llm(profile, context, **llm_kwargs), "llm"
        except Exception as exc:
            return f"> No se pudo usar el LLM ({exc}).\n\n{describe_variables_rules(profile)}", "reglas"
    return describe_variables_rules(profile), "reglas"


def _fmt(v, nd: int = 2) -> str:
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def generate_rule_based_report(profile: Profile) -> str:
    ov = profile.overview
    mo = profile.missing_overview
    lines = [f"# Informe exploratorio: {profile.dataset_name}", ""]

    lines += ["## Resumen general", ""]
    tipos = ", ".join(f"{n} {t}{'s' if n != 1 else ''}" for t, n in ov["tipos"].items())
    lines.append(f"El conjunto tiene **{ov['filas']:,} filas** y **{ov['columnas']} columnas** ({tipos}).")
    if ov["filas_duplicadas"]:
        lines.append(f"Se encontraron **{ov['filas_duplicadas']} filas duplicadas**.")
    if profile.cleaning_actions:
        lines += ["", "Durante la carga se hicieron estos ajustes:", ""]
        lines += [f"- {a}" for a in profile.cleaning_actions]
    lines.append("")

    lines += ["## Variables y su importancia", "", describe_variables_rules(profile), ""]
    if profile.target and not profile.target_importance.empty:
        lines += [f"Asociación con la variable objetivo **{profile.target}**:", ""]
        for col, r in profile.target_importance.head(10).iterrows():
            lines.append(f"- **{col}**: {r['asociación']} ({r['medida']})")
        lines.append("")

    lines += ["## Calidad de los datos", "", "### Valores faltantes", ""]
    if mo["celdas_faltantes"] == 0:
        lines.append("No hay valores faltantes.")
    else:
        lines.append(
            f"Faltan {mo['celdas_faltantes']:,} celdas ({mo['pct_celdas']}% del total), repartidas en "
            f"{mo['columnas_con_faltantes']} columnas; {mo['filas_con_faltantes']:,} filas tienen al menos un faltante."
        )
        lines.append("")
        for col, row in profile.missing[profile.missing["faltantes"] > 0].iterrows():
            lines.append(f"- **{col}**: {row['faltantes']} ({row['pct']}%). Sugerencia: {row['sugerencia'][0].lower() + row['sugerencia'][1:]}.")
    lines += ["", "### Valores atípicos", ""]
    out = profile.outliers
    flagged = out[out["atípicos"] > 0] if not out.empty else out
    if flagged.empty:
        lines.append(f"No se detectaron atípicos con el método {profile.outlier_method.upper()} (k = {profile.outlier_k}).")
    else:
        lines.append(f"Método {profile.outlier_method.upper()} con k = {profile.outlier_k}:")
        lines.append("")
        for col, row in flagged.iterrows():
            lines.append(
                f"- **{col}**: {row['atípicos']} {'atípico' if row['atípicos'] == 1 else 'atípicos'} ({row['pct']}%), fuera de "
                f"[{_fmt(row['límite_inf'])}, {_fmt(row['límite_sup'])}]; {row['alto']} altos y {row['bajo']} bajos."
            )
    lines.append("")

    lines += ["## Hallazgos principales", ""]
    if not profile.numeric.empty:
        for col, r in profile.numeric.iterrows():
            shape = "simétrica"
            if abs(r["asimetría"]) > 1:
                shape = "con cola a la derecha" if r["asimetría"] > 0 else "con cola a la izquierda"
            lines.append(
                f"- **{col}**: media {_fmt(r['media'])}, mediana {_fmt(r['mediana'])}, "
                f"rango [{_fmt(r['mín'])}, {_fmt(r['máx'])}]; distribución {shape}."
            )
    if not profile.categorical.empty:
        for col, r in profile.categorical.iterrows():
            lines.append(
                f"- **{col}**: {r['únicos']} categorías; la más frecuente es «{r['más_frecuente']}» "
                f"({r['pct_más_frecuente']}%)."
            )
    if not profile.datetime.empty:
        for col, r in profile.datetime.iterrows():
            lines.append(f"- **{col}**: de {r['mín']:%Y-%m-%d} a {r['máx']:%Y-%m-%d} ({r['rango_días']} días).")
    lines.append("")

    lines += ["## Relaciones entre variables", ""]
    if profile.strong_pairs:
        for p in profile.strong_pairs[:10]:
            sentido = "positiva" if p["r"] > 0 else "negativa"
            lines.append(f"- **{p['var_1']}** y **{p['var_2']}**: correlación {sentido} (r = {p['r']}).")
        lines += ["", "Una correlación alta no implica que una variable cause la otra."]
    else:
        lines.append("No hay pares de variables numéricas con correlación fuerte (|r| ≥ 0.5).")
    lines.append("")

    lines += ["## Recomendaciones", ""]
    recs = [a for a in profile.alerts if "duplicadas" not in a]
    if ov["filas_duplicadas"]:
        recs.append(f"Revisar las {ov['filas_duplicadas']} filas duplicadas y eliminarlas si no son registros legítimos.")
    if any("asimétrica" in a for a in recs):
        recs.append("Para las variables muy asimétricas, preferir mediana y rango intercuartílico o una transformación logarítmica.")
    if not flagged.empty:
        recs.append("Verificar los atípicos: corregir si son errores de captura, conservar si son casos reales.")
    lines += [f"- {r}" for r in recs] or ["- Los datos se ven consistentes; no hay alertas destacadas."]
    lines += ["", "_Informe generado automáticamente sin LLM (configura Groq u Ollama para un informe redactado por IA)._"]
    return "\n".join(lines)


def generate_report(profile: Profile, use_llm: bool = True, context: str = "", **llm_kwargs) -> tuple[str, str]:
    """Devuelve (informe, origen). El origen es 'llm' o 'reglas'.

    llm_kwargs se pasan a llm.chat: provider, model, api_key.
    """
    provider = llm_kwargs.get("provider") or llm.default_provider()
    if use_llm and llm.is_available(provider, llm_kwargs.get("api_key")):
        try:
            return generate_llm_report(profile, context, **llm_kwargs), "llm"
        except Exception as exc:  # red, cuota, modelo retirado...
            fallback = generate_rule_based_report(profile)
            return f"> No se pudo usar el LLM ({exc}). Se muestra el informe basado en reglas.\n\n{fallback}", "reglas"
    return generate_rule_based_report(profile), "reglas"
