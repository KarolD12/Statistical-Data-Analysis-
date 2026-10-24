# 📊 EDA asistido por LLM

Sube un CSV y obtén, en segundos, un análisis exploratorio completo: estadísticas
descriptivas, valores faltantes, valores atípicos, correlaciones y un **informe en
lenguaje natural redactado por un LLM**, en la nube con [Groq](https://groq.com) o
en tu propio computador con [Ollama](https://ollama.com).

**Stack:** Python · pandas · Plotly · Streamlit · Groq / Ollama

---

## ¿Por qué hace falta limpiar un CSV?

Un CSV es texto plano: cada línea es una fila y los valores se separan con comas
(o con punto y coma en países donde la coma es el separador decimal). Es simple y
universal, pero **no declara tipos de datos, no define qué es un valor vacío y
puede venir con separadores, codificaciones o encabezados inconsistentes**. Por eso,
antes de analizar, el sistema:

| Problema | Qué hace el sistema |
|---|---|
| Codificación desconocida | Prueba `utf-8`, `utf-8-sig`, `cp1252` y `latin-1` (tildes y ñ correctas) |
| Separador desconocido | Detecta `,` `;` tabulador o `\|` |
| Coma decimal (`14,5`) | La detecta cuando el separador no es coma y convierte a número |
| Vacíos con muchas formas | `""`, `NA`, `N/A`, `null`, `-`, `?`, `sin dato`, `#N/A`… → faltante |
| Todo llega como texto | Infiere columnas numéricas (también con `$`, `%`) y de fecha |
| Filas/columnas vacías | Las elimina y lo informa |

Cada decisión queda registrada y aparece en el informe.

## Funcionalidades

- **Variables y su importancia** (primera pestaña): el rol de cada columna
  (medida, categoría, fecha, identificador, constante…), un **puntaje de
  información** de 0 a 100 y, con LLM, una descripción de qué representa cada
  variable y por qué importa. Si eliges una **variable objetivo**, se mide qué
  tanto se relaciona cada columna con ella (|Spearman|, η o V de Cramér según
  los tipos).
- **Estadísticas descriptivas** por tipo: numéricas (media, mediana, cuartiles,
  asimetría, curtosis, coeficiente de variación), categóricas (únicos, moda,
  frecuencia) y fechas (rango).
- **Valores faltantes**: conteo y porcentaje por columna, mapa de faltantes y
  sugerencia de tratamiento (media, mediana, moda o eliminar).
- **Valores atípicos** con tres métodos configurables:
  - **IQR** (Tukey): fuera de `[Q1 − k·IQR, Q3 + k·IQR]`, k = 1.5 por defecto.
  - **Puntuación z**: `|x − μ| / σ > k`, k = 3.
  - **z modificado (MAD)**: robusto a los propios atípicos, k = 3.5.
- **Correlaciones** de Pearson con mapa de calor y lista de pares fuertes.
- **Alertas** automáticas: duplicados, columnas constantes, identificadores,
  distribuciones muy asimétricas, negativos sospechosos.
- **Informe con IA**: el perfil estadístico (no los datos crudos) se envía al
  modelo, que redacta un informe estructurado en español. Se descarga en Markdown.
- **Dos proveedores de LLM**: Groq (nube, rápido, gratis con límites) u Ollama
  (local, privado, sin internet).
- **Funciona sin LLM**: si no hay ninguno disponible o falla la llamada, genera
  un informe basado en reglas.

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/KarolD12/Statistical-Data-Analysis-.git
cd Statistical-Data-Analysis-
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Configura tu clave de Groq (gratis en [console.groq.com/keys](https://console.groq.com/keys)):

```bash
cp .env.example .env      # y edita GROQ_API_KEY
```

También puedes pegar la clave directamente en la barra lateral de la app.

### Opción local con Ollama (sin internet ni API key)

1. Instala Ollama desde [ollama.com/download](https://ollama.com/download)
   (en Windows: `winget install Ollama.Ollama`).
2. Descarga un modelo pequeño:

   ```bash
   ollama pull llama3.2:3b
   ```

3. En la app elige **Ollama (local)** en la barra lateral, o pon
   `LLM_PROVIDER=ollama` en `.env`.

| Modelo | Tamaño | RAM recomendada | Comentario |
|---|---|---|---|
| `llama3.2:1b` | 1.3 GB | 4 GB | Muy rápido, redacción básica |
| `llama3.2:3b` | 2.0 GB | 8 GB | **Recomendado** para equipos sin GPU |
| `qwen2.5:3b` | 1.9 GB | 8 GB | Buen español y buen manejo de tablas |
| `gemma3:4b` | 3.3 GB | 8-16 GB | Mejor redacción, más lento |

Sin GPU, un modelo de 3B redacta el informe en unos minutos; con Groq tarda
segundos. Los modelos pequeños siguen peor las instrucciones y a veces se
equivocan con los números: revisa el informe contra las tablas.

## Uso

### Aplicación web

```bash
streamlit run app.py
```

Abre `http://localhost:8501`, sube un CSV (o usa los datos de ejemplo) y recorre las
pestañas: *Vista previa*, *Estadísticas*, *Faltantes*, *Atípicos*,
*Correlaciones* e *Informe IA*.

### Línea de comandos

```bash
python -m eda data/ventas_ejemplo.csv -o informe.md --html informe.html
python -m eda datos.csv --method mad --context "Ventas diarias de 2025"
python -m eda datos.csv --target satisfaccion
python -m eda datos.csv --provider ollama --model llama3.2:3b
python -m eda datos.csv --no-llm          # sin LLM
```

### Como librería

```python
from eda import load_csv, profile_dataframe, generate_report

loaded = load_csv("datos.csv")
profile = profile_dataframe(loaded.df, outlier_method="iqr", cleaning_actions=loaded.actions)
informe, origen = generate_report(profile)   # origen: "llm" o "reglas"
```

## Estructura

```
eda-llm/
├── app.py                  # Interfaz Streamlit
├── eda/
│   ├── loader.py           # Lectura y limpieza robusta del CSV
│   ├── stats.py            # Estadísticas descriptivas y correlaciones
│   ├── quality.py          # Faltantes, atípicos y alertas
│   ├── importance.py       # Rol, puntaje de información e importancia de variables
│   ├── llm.py              # Cliente para Groq y Ollama
│   ├── profile.py          # Reúne el análisis en un objeto Profile
│   ├── plots.py            # Gráficos Plotly
│   ├── report.py           # Prompts, informe con LLM e informe por reglas
│   └── __main__.py         # CLI
├── data/ventas_ejemplo.csv # Datos sintéticos con faltantes, atípicos y duplicados
├── scripts/generar_ejemplo.py
└── tests/
```

## Cómo funciona el informe con IA

1. Se calcula el perfil estadístico (`Profile`).
2. Se serializa a un JSON compacto: resúmenes, faltantes, atípicos, correlaciones
   fuertes y alertas. **Los datos fila por fila no se envían al modelo**; en la app
   puedes ver exactamente qué se envía en *Ver los datos que recibe el modelo*.
   Para describir las variables se envían también hasta 3 valores de ejemplo por
   columna (las categorías más frecuentes, o mínimo, mediana y máximo), para que
   el modelo pueda inferir qué representa cada una. Si tus datos son sensibles,
   usa Ollama: nada sale de tu computador.
3. Un *system prompt* le pide al modelo usar solo esos números, no inventar,
   advertir que correlación no es causalidad y seguir una estructura fija.
4. Modelos por defecto: `openai/gpt-oss-120b` en Groq y `llama3.2:3b` en Ollama;
   se cambian con variables de entorno o desde la app.

## Configuración

| Variable | Descripción | Por defecto |
|---|---|---|
| `LLM_PROVIDER` | `groq` u `ollama` | `groq` |
| `GROQ_API_KEY` | Clave de la API de Groq | — |
| `GROQ_MODEL` | Modelo de Groq | `openai/gpt-oss-120b` |
| `OLLAMA_MODEL` | Modelo local | `llama3.2:3b` |
| `OLLAMA_BASE_URL` | Dirección del servidor Ollama | `http://localhost:11434` |

En Streamlit Community Cloud, define estas variables en *Settings → Secrets*.

## Pruebas

```bash
pip install pytest
pytest
```

## Licencia

[MIT](LICENSE)
