from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_with_sample_data(monkeypatch):
    # Vacía (no eliminada) para que load_dotenv no la cargue desde .env.
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 5
    at.button[0].click().run()
    assert not at.exception
    assert any("Informe exploratorio" in m.value for m in at.markdown)
