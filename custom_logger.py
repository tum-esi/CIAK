from loguru import logger
import sys
from pathlib import Path
from typing import Optional

def setup_logging():
    """
    Custom Logger
    
    """
    logger.remove()
    logger.level("TRACE",    color="<cyan>",             icon="🔍")
    logger.level("DEBUG",    color="<blue>",             icon="🐛")
    logger.level("INFO",     color="<blue>",            icon="ℹ️")
    logger.level("SUCCESS",  color="<green>",            icon="✅")
    logger.level("WARNING",  color="<yellow>",           icon="⚠️")
    logger.level("ERROR",    color="<red>",              icon="❌")
    logger.level("CRITICAL", color="<bold><RED>",        icon="💥")

    console_fmt = (
        "{level.icon} <level>{level: <8}</level> | "
        "<magenta>{function}</magenta>:<magenta>{line}</magenta> — "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        colorize=True,
        backtrace=True,   
        diagnose=True,    
        format=console_fmt,
    )

    return logger
