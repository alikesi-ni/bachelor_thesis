import logging
import sys

class LoggerFactory:
    _formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    @staticmethod
    def _get_log_level(level_str: str) -> int:
        level_mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }
        return level_mapping.get(level_str.lower(), logging.INFO)

    @staticmethod
    def get_full_logger(name: str, log_path: str = None, level="info") -> logging.Logger:
        logger = logging.getLogger(name)

        log_level = LoggerFactory._get_log_level(level)
        logger.setLevel(log_level)
        logger.handlers.clear()

        # File handler
        if log_path:
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setFormatter(LoggerFactory._formatter)
            logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(LoggerFactory._formatter)
        logger.addHandler(console_handler)

        return logger


    @staticmethod
    def get_console_logger(name: str, level="info") -> logging.Logger:
        logger = logging.getLogger(name)

        log_level = LoggerFactory._get_log_level(level)
        logger.setLevel(log_level)
        logger.handlers.clear()

        # Console handler only
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(LoggerFactory._formatter)
        logger.addHandler(console_handler)

        return logger
