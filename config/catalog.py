from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TEMPORARY_DATA_DIR = DATA_DIR / "temporary"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = PROJ_ROOT / "logs"
