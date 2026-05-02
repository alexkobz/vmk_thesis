from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

from config.catalog import PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, TEMPORARY_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR

# Load environment variables from .env file if it exists
load_dotenv()
logger.add(
    LOG_DIR / f"run_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log",
    level="INFO",
    rotation="10 MB",
    retention="14 days",
    compression="zip",
)
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

__all__ = [
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "TEMPORARY_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "logger",
]
