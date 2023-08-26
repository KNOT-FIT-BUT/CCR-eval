import logging
import sys

LOGGING_LEVEL = logging.DEBUG
LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'

METRICS_AT_K = [5, 10, 20, 50, 100]
INDEX_TYPES = ["bm25", "colbert"]

# Anything above this considered relevant
RELEVANCE_THRESHOLD = 0.0 

# Path to morpho model (for lemmatization)
MORPHODITA_MODEL = "morphodita/czech-morfflex2.0-pdtc1.0-220710/czech-morfflex2.0-pdtc1.0-220710.tagger"

# Set up logger object
logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter(LOGGING_FORMAT)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(LOGGING_LEVEL)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(LOGGING_LEVEL)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)