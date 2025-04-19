import logging
import sys

# Set up logger
logger = logging.getLogger("demo_logger")
logger.setLevel(logging.DEBUG)  # Try changing this to DEBUG, WARNING, ERROR, etc.
logger.handlers.clear()

# Formatter for output
formatter = logging.Formatter("%(levelname)s — %(message)s")

# Console handler
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Try logging at different levels
logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")
logger.error("This is an ERROR message")
logger.critical("This is a CRITICAL message")
