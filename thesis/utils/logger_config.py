import logging
import sys

# Global variable to store current log file path
_log_file_path = None

def setup_logger(name: str, log_path: str = None) -> logging.Logger:
    global _log_file_path

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Set or update the log file path
    if log_path:
        _log_file_path = log_path
    elif not _log_file_path:
        _log_file_path = "default.log"  # Fallback if never set

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-5s - %(message)s",  # ✅ Logging format placeholders
        datefmt="%Y-%m-%d %H:%M:%S"  # ✅ Time formatting only here
    )

    # File handler
    file_handler = logging.FileHandler(_log_file_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def change_log_file(new_path: str):
    """
    Change the log file for all loggers dynamically.
    """
    global _log_file_path
    _log_file_path = new_path

    # Update all existing loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

        file_handler = logging.FileHandler(_log_file_path, mode="a")
        formatter = logging.Formatter(
            "%Y-%m-%d %H:%M:%S %(levelname)-5s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)