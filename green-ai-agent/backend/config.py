from pathlib import Path

# Models
LIGHT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
HEAVY_MODEL = "textattack/bert-base-uncased-SST-2"

# Policy
DEFAULT_THRESHOLD = 0.85
MAX_CHARS = 2000  # guard against huge inputs

# Paths
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "green_metrics.sqlite3"
SAMPLES_PATH = ROOT / "data" / "samples.txt"
