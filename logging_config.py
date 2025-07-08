# logging_config.py
import logging, sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    fmt = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
    )
