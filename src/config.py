from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ajuste os nomes conforme seus arquivos reais
BASE_01_PATH = DATA_RAW / "Base 1 – Internações Hospitalares.csv"
BASE_02_PATH = DATA_RAW / "Base 2 – Itens da Internação.csv"

OUTPUT_DATASET_MODEL = DATA_PROCESSED / "dataset_modelagem.csv"
OUTPUT_SCORE = DATA_PROCESSED / "score_risco_internacoes.csv"
OUTPUT_MODEL = MODELS_DIR / "model_risco_internacao.joblib"


@dataclass(frozen=True)
class TargetConfig:
    p90_q: float = 0.90  # percentil para proxy