"""EDA asistido por LLM: carga un CSV, lo perfila y redacta un informe."""

from .loader import LoadResult, load_csv
from .profile import Profile, profile_dataframe
from .report import generate_report

__all__ = ["LoadResult", "Profile", "generate_report", "load_csv", "profile_dataframe"]
__version__ = "0.1.0"
