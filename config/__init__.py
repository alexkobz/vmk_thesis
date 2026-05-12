from dotenv import load_dotenv

from config.catalog import PROJ_ROOT, CONFIG_DIR, DATA_DIR, RAW_DATA_DIR, TEMPORARY_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR

load_dotenv()

__all__ = [
    "PROJ_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "TEMPORARY_DATA_DIR",
    "PROCESSED_DATA_DIR",
]
