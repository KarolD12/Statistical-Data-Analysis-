"""Interfaz web: streamlit run app.py"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from eda import llm, plots
from eda.loader import load_csv
from eda.profile import profile_dataframe
from eda.quality import OUTLIER_METHODS, default_k, outlier_mask
from eda.report import describe_variables, generate_report

load_dotenv()

SAMPLE = Path(__file__).parent / "data" / "ventas_ejemplo.csv"
MAX_MB = 50
NO_TARGET = "— Ninguna —"

st.set_page_config(page_title="EDA asistido por LLM", page_icon="📊", layout="wide")


def secret(name: str) -> str | None:
    try:
        return st.secrets.get(name) or os.getenv(name)
    except Exception:  # no hay secrets.toml
        return os.getenv(name)


@st.cache_data(show_spinner="Leyendo el CSV…")
def cached_load(raw: bytes):
    return load_csv(raw, max_bytes=MAX_MB * 1024 * 1024)


@st.cache_data(show_spinner="Calculando estadísticas…")
def cached_profile(raw: bytes, name: str, method: str, k: float, target: str | None):
    loaded = cached_load(raw)
    return profile_dataframe(loaded.df, method, k, cleaning_actions=loaded.actions, dataset_name=name,
                             target=target)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("1. Datos")
    uploaded = st.file_uploader("Sube un archivo CSV", type=["csv", "txt", "tsv"])
    use_sample = st.toggle("Usar datos de ejemplo", value=uploaded is None)
    context = st.text_area("Contexto (opcional)", placeholder="Ej.: ventas diarias de una cadena de cafeterías en 2025.",
                           help="Ayuda al modelo a interpretar las variables.")

    st.header("2. Atípicos")
    method_labels = {"iqr": "Rango intercuartílico (IQR)", "zscore": "Puntuación z", "mad": "z modificado (MAD)"}
    method = st.selectbox("Método", OUTLIER_METHODS, format_func=method_labels.get)
    k = st.number_input("Umbral k", min_value=0.5, max_value=10.0, value=default_k(method), step=0.5,
                        help="IQR: 1.5 habitual, 3 para extremos. z: 3. MAD: 3.5.")

    st.header("3. Modelo de lenguaje")
    provider_labels = {"groq": "Groq (nube, rápido)", "ollama": "Ollama (local, privado)"}
    default_prov = secret("LLM_PROVIDER") or "groq"
    provider = st.radio("Proveedor", llm.PROVIDERS, format_func=provider_labels.get,
                        index=llm.PROVIDERS.index(default_prov) if default_prov in llm.PROVIDERS else 0)
    api_key = None
    if provider == "groq":
        env_key = secret("GROQ_API_KEY")
        api_key = st.text_input("API key de Groq", type="password", value="",
                                placeholder="Configurada en el entorno" if env_key else "gsk_…",
                                help="Consíguela gratis en console.groq.com. No se guarda.") or env_key
        default_model = secret("GROQ_MODEL") or llm.GROQ_DEFAULT
        models = llm.GROQ_MODELS if default_model in llm.GROQ_MODELS else [default_model, *llm.GROQ_MODELS]
        model = st.selectbox("Modelo", models, index=models.index(default_model))
        llm_ready = bool(api_key)
    else:
        installed = llm.ollama_models()
        if installed:
            default_model = secret("OLLAMA_MODEL") or llm.OLLAMA_DEFAULT
            model = st.selectbox("Modelo local", installed,
                                 index=installed.index(default_model) if default_model in installed else 0)
            st.caption("En CPU, el informe completo puede tardar varios minutos.")
        else:
            model = None
            st.warning("Ollama no responde o no tiene modelos. Instálalo desde ollama.com y ejecuta "
                       f"`ollama pull {llm.OLLAMA_DEFAULT}`.")
        llm_ready = bool(installed)

llm_kwargs = {"provider": provider, "model": model, "api_key": api_key}
llm_label = f"{provider_labels[provider].split(' (')[0]} · {model}"

if uploaded is not None and not use_sample:
    raw, name = uploaded.getvalue(), uploaded.name
elif use_sample and SAMPLE.exists():
    raw, name = SAMPLE.read_bytes(), SAMPLE.name
else:
    st.title("📊 Análisis exploratorio asistido por LLM")
    st.info("Sube un CSV en la barra lateral para empezar.")
    st.stop()

try:
    loaded = cached_load(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

df = loaded.df

st.title("📊 Análisis exploratorio asistido por LLM")
st.caption(f"Archivo: **{name}**")

target_choice = st.sidebar.selectbox(
    "Variable objetivo (opcional)", [NO_TARGET, *df.columns],
    help="Si eliges una, se mide qué tanto se relaciona cada variable con ella.",
)
target = None if target_choice == NO_TARGET else target_choice
profile = cached_profile(raw, name, method, k, target)
ov, mo = profile.overview, profile.missing_overview

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Filas", f"{ov['filas']:,}")
c2.metric("Columnas", ov["columnas"])
c3.metric("Celdas faltantes", f"{mo['pct_celdas']}%")
n_out = int(profile.outliers["atípicos"].sum()) if not profile.outliers.empty else 0
c4.metric("Valores atípicos", f"{n_out:,}")
c5.metric("Filas duplicadas", ov["filas_duplicadas"])

tabs = st.tabs(["Variables", "Vista previa", "Estadísticas", "Faltantes", "Atípicos", "Correlaciones", "Informe IA"])

with tabs[0]:
    st.subheader("Las variables del conjunto y su importancia")
    # Groq responde en segundos: se describe automáticamente. Un modelo local en CPU tarda,
    # así que ahí se pide con un botón.
    key = (name, provider, model, context, target)
    cached = st.session_state.get("variables")
    if cached is None or cached[0] != key:
        cached = None
        if not llm_ready or provider == "groq":
            with st.spinner("Describiendo las variables…"):
                text, source = describe_variables(profile, use_llm=llm_ready, context=context, **llm_kwargs)
            cached = (key, text, source)
            st.session_state["variables"] = cached
    if cached is None:
        st.info("Pulsa el botón para que el modelo local describa cada variable.")
        if st.button("Describir variables con IA", type="primary"):
            with st.spinner(f"Describiendo las variables con {model}…"):
                text, source = describe_variables(profile, context=context, **llm_kwargs)
            st.session_state["variables"] = (key, text, source)
            st.rerun()
    else:
        _, text, source = cached
        st.caption("Descripción por " + (llm_label if source == "llm" else "reglas (sin LLM)"))
        st.markdown(text)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plots.score_bar(profile.variables["puntaje_info"], "Puntaje de información (0-100)",
                                        "Puntaje", max_value=100), width="stretch")
    with right:
        if target and not profile.target_importance.empty:
            st.plotly_chart(plots.score_bar(profile.target_importance["asociación"],
                                            f"Asociación con {target}", "Asociación (0-1)", max_value=1),
                            width="stretch")
        else:
            st.info("Elige una **variable objetivo** en la barra lateral para ver qué variables se relacionan "
                    "más con ella.")
    with st.expander("Cómo se calcula"):
        st.markdown(
            "- **Puntaje de información** = completitud × variabilidad × (0.6 + 0.4 × asociación máxima con "
            "otra variable). Identificadores, constantes y texto libre puntúan 0.\n"
            "- **Variabilidad**: entropía normalizada de las categorías o del histograma (percentil 1-99).\n"
            "- **Asociación** (0 a 1): |Spearman| entre numéricas, razón de correlación η entre categórica y "
            "numérica, V de Cramér entre categóricas.\n"
            "- Es una guía exploratoria, no un modelo predictivo: una asociación alta no implica causalidad."
        )
        st.dataframe(profile.variables, width="stretch")

with tabs[1]:
    st.dataframe(df.head(200), width="stretch")
    with st.expander("Limpieza aplicada al cargar", expanded=True):
        for action in loaded.actions:
            st.markdown(f"- {action}")
    if profile.alerts:
        st.subheader("Alertas")
        for alert in profile.alerts:
            st.warning(alert, icon="⚠️")

with tabs[2]:
    if not profile.numeric.empty:
        st.subheader("Variables numéricas")
        st.dataframe(profile.numeric, width="stretch")
        col = st.selectbox("Distribución de", profile.numeric.index, key="hist")
        st.plotly_chart(plots.histogram(df, col), width="stretch")
    if not profile.categorical.empty:
        st.subheader("Variables categóricas")
        st.dataframe(profile.categorical, width="stretch")
        col = st.selectbox("Frecuencias de", profile.categorical.index, key="cat")
        st.plotly_chart(plots.category_bar(df, col), width="stretch")
    if not profile.datetime.empty:
        st.subheader("Fechas")
        st.dataframe(profile.datetime, width="stretch")

with tabs[3]:
    if mo["celdas_faltantes"] == 0:
        st.success("No hay valores faltantes.")
    else:
        st.markdown(
            f"**{mo['celdas_faltantes']:,}** celdas faltantes en **{mo['columnas_con_faltantes']}** columnas; "
            f"**{mo['filas_con_faltantes']:,}** filas tienen al menos uno."
        )
        st.plotly_chart(plots.missing_bar(df), width="stretch")
        st.plotly_chart(plots.missing_matrix(df), width="stretch")
        st.dataframe(profile.missing[profile.missing["faltantes"] > 0], width="stretch")

with tabs[4]:
    out = profile.outliers
    if out.empty:
        st.info("No hay columnas numéricas con suficientes valores distintos para buscar atípicos.")
    else:
        st.dataframe(out.drop(columns="ejemplos"), width="stretch")
        col = st.selectbox("Ver columna", out.index, key="box")
        st.plotly_chart(plots.box_with_outliers(df, col, method, k), width="stretch")
        rows = df[outlier_mask(df, col, method, k)]
        if not rows.empty:
            st.caption(f"Filas con atípicos en {col}")
            st.dataframe(rows, width="stretch")

with tabs[5]:
    fig = plots.correlation_heatmap(profile.correlation)
    if fig is None:
        st.info("Se necesitan al menos dos columnas numéricas no constantes.")
    else:
        st.plotly_chart(fig, width="stretch")
        if profile.strong_pairs:
            st.dataframe(pd.DataFrame(profile.strong_pairs), width="stretch", hide_index=True)

with tabs[6]:
    if not llm_ready:
        st.info("Sin un modelo disponible se genera un informe basado en reglas.")
    if st.button("Generar informe", type="primary"):
        with st.spinner(f"Redactando el informe con {model}…" if llm_ready else "Generando el informe…"):
            text, source = generate_report(profile, use_llm=llm_ready, context=context, **llm_kwargs)
        st.session_state["report"] = (name, text, source)
    if st.session_state.get("report", (None,))[0] == name:
        _, text, source = st.session_state["report"]
        st.caption("Redactado por " + (llm_label if source == "llm" else "reglas (sin LLM)"))
        st.markdown(text)
        st.download_button("Descargar informe (.md)", text, file_name=f"informe_{Path(name).stem}.md",
                           mime="text/markdown")
    with st.expander("Ver los datos que recibe el modelo"):
        st.code(profile.to_json(), language="json")
