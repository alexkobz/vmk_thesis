from dotenv import load_dotenv
from loguru import logger

from config.catalog import PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, TEMPORARY_DATA_DIR, PROCESSED_DATA_DIR

# Load environment variables from .env file if it exists
load_dotenv()
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

__all__ = [
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "TEMPORARY_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "logger",
]
