"""
logger.py — Logging configuration for FYP experiments.

Sets up a logger that writes to both the console and a timestamped log file.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "fyp",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger with file + optional console output.

    Parameters
    ----------
    name    : logger name
    log_dir : directory to write log files
    level   : logging level (INFO, DEBUG, etc.)
    console : whether to also print to stdout

    Returns
    -------
    logging.Logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.propagate = False
    logger.info(f"Log file: {log_file}")
    return logger