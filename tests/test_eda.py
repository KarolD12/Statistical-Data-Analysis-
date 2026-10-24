import numpy as np
import pandas as pd
import pytest

from eda import plots
from eda.loader import load_csv
from eda.profile import profile_dataframe
from eda.quality import detect_outliers, missing_overview, quality_alerts
from eda.report import build_prompt, generate_report


def test_comma_csv_with_na_tokens():
    raw = b"producto,cantidad,precio\nCafe,3,12500\nArepa,10,3000\nJugo,,4500\nTe,N/A,sin dato\n"
    res = load_csv(raw)
    assert res.separator == ","
    assert pd.api.types.is_numeric_dtype(res.df["cantidad"])
    assert res.df["cantidad"].isna().sum() == 2
    assert res.df["precio"].isna().sum() == 1


def test_semicolon_decimal_comma_latin1():
    raw = "ciudad;temperatura;fecha\nBogotá;14,5;03/01/2025\nMedellín;24,1;15/02/2025\nCali;;28/02/2025\n"
    res = load_csv(raw.encode("cp1252"))
    assert res.separator == ";"
    assert res.decimal == ","
    assert res.df["ciudad"].iloc[0] == "Bogotá"
    assert res.df["temperatura"].iloc[0] == pytest.approx(14.5)
    assert pd.api.types.is_datetime64_any_dtype(res.df["fecha"])
    assert res.df["fecha"].iloc[1].month == 2  # día primero


def test_drops_empty_rows_and_columns():
    raw = b"a,b,vacia\n1,x,\n,,\n2,y,\n"
    res = load_csv(raw)
    assert list(res.df.columns) == ["a", "b"]
    assert len(res.df) == 2


def test_empty_file_raises():
    with pytest.raises(ValueError):
        load_csv(b"   ")


def test_outliers_iqr_detects_extreme():
    df = pd.DataFrame({"x": list(range(1, 21)) + [500], "flag": [0, 1] * 10 + [0]})
    out = detect_outliers(df, "iqr")
    assert "flag" not in out.index  # binaria: se omite
    assert out.loc["x", "atípicos"] == 1
    assert out.loc["x", "alto"] == 1


@pytest.mark.parametrize("method", ["iqr", "zscore", "mad"])
def test_outlier_methods_run(method):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": np.append(rng.normal(0, 1, 200), [15, -12])})
    out = detect_outliers(df, method)
    assert out.loc["v", "atípicos"] >= 2


def test_missing_and_alerts():
    df = pd.DataFrame({"a": [1, None, None, None], "c": ["k"] * 4})
    mo = missing_overview(df)
    assert mo["celdas_faltantes"] == 3
    alerts = " ".join(quality_alerts(df))
    assert "'a'" in alerts and "constante" in alerts


def test_profile_report_and_prompt():
    df = pd.DataFrame({
        "x": np.arange(50, dtype=float),
        "y": np.arange(50) * 2.0 + 1,
        "cat": ["a", "b"] * 25,
    })
    df.loc[3, "x"] = np.nan
    prof = profile_dataframe(df, dataset_name="prueba.csv")
    assert prof.strong_pairs[0]["r"] > 0.99
    report, source = generate_report(prof, use_llm=False)
    assert source == "reglas"
    assert "### Valores faltantes" in report
    assert "**x**: 1 (2.0%)" in report
    prompt = build_prompt(prof, context="datos de prueba")
    assert "prueba.csv" in prompt and "datos de prueba" in prompt


def test_plots_build(tmp_path):
    raw = b"a,b,c\n1,2,x\n2,,y\n3,6,x\n100,8,z\n5,10,x\n6,12,y\n"
    df = load_csv(raw).df
    prof = profile_dataframe(df)
    figs = [
        plots.histogram(df, "a"),
        plots.box_with_outliers(df, "a"),
        plots.missing_bar(df),
        plots.missing_matrix(df),
        plots.correlation_heatmap(prof.correlation),
        plots.category_bar(df, "c"),
    ]
    assert all(f is not None for f in figs)


def test_llm_path_with_mocked_groq(monkeypatch):
    import groq

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = type("M", (), {"content": "## Resumen general\nTodo bien."})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    class FakeGroq:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr(groq, "Groq", FakeGroq)
    prof = profile_dataframe(pd.DataFrame({"x": range(10)}))
    report, source = generate_report(prof, api_key="gsk_test", model="openai/gpt-oss-120b")
    assert source == "llm"
    assert report.startswith("## Resumen general")
    assert captured["api_key"] == "gsk_test"
    assert captured["messages"][0]["role"] == "system"


def test_llm_failure_falls_back(monkeypatch):
    import groq

    def boom(api_key):
        raise ConnectionError("sin red")

    monkeypatch.setattr(groq, "Groq", boom)
    prof = profile_dataframe(pd.DataFrame({"x": range(10)}))
    report, source = generate_report(prof, api_key="gsk_test")
    assert source == "reglas"
    assert "sin red" in report


def test_variable_importance_roles_and_target():
    from eda.importance import target_importance, variable_overview

    rng = np.random.default_rng(1)
    n = 200
    grupo = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame({
        "id": np.arange(n),
        "constante": ["x"] * n,
        "grupo": grupo,
        "ruido": rng.normal(size=n),
        "y": np.where(grupo == "a", 10.0, 0.0) + rng.normal(0, 1, n),
    })
    v = variable_overview(df)
    assert v.loc["id", "rol"] == "identificador" and v.loc["id", "puntaje_info"] == 0
    assert v.loc["constante", "rol"] == "constante"
    assert v.loc["grupo", "más_relacionada_con"] == "y"
    imp = target_importance(df, "y")
    assert imp.index[0] == "grupo" and imp.loc["grupo", "medida"] == "η"
    assert imp.loc["grupo", "asociación"] > 0.8 > imp.loc["ruido", "asociación"]


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        profile_dataframe(pd.DataFrame({"x": range(10)}), target="no_existe")


def test_ollama_provider_mocked(monkeypatch):
    import httpx

    from eda import llm
    from eda.report import describe_variables

    sent = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "<think>pienso</think>| Variable | Qué representa |"}}

    def fake_post(url, json, timeout):
        sent["url"], sent["json"] = url, json
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llm, "ollama_models", lambda base_url=None: ["llama3.2:3b"])
    prof = profile_dataframe(pd.DataFrame({"x": range(30), "c": ["a", "b", "c"] * 10}))
    text, source = describe_variables(prof, provider="ollama", model="llama3.2:3b")
    assert source == "llm"
    assert text == "| Variable | Qué representa |"  # sin el bloque <think>
    assert sent["url"].endswith("/api/chat")
    assert sent["json"]["options"]["num_ctx"] == 8192
