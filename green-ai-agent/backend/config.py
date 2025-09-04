# from pathlib import Path

# # Models
# LIGHT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# HEAVY_MODEL = "textattack/bert-base-uncased-SST-2"

# # Policy
# DEFAULT_THRESHOLD = 0.85
# MAX_CHARS = 2000  # guard against huge inputs

# # Paths
# ROOT = Path(__file__).resolve().parents[1]
# DB_PATH = ROOT / "green_metrics.sqlite3"
# SAMPLES_PATH = ROOT / "data" / "samples.txt"



from pathlib import Path

# Email Classification Models
LIGHT_MODEL = "distilbert-base-uncased"  # Will fine-tune for email
HEAVY_MODEL = "bert-base-uncased"        # Will fine-tune for email

# Categories
EMAIL_CATEGORIES = ["work", "spam", "promotions", "personal", "support", "newsletter"]
NUM_CLASSES = len(EMAIL_CATEGORIES)

# Energy & Performance Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
ENERGY_SAVING_THRESHOLD = 0.30  # 30% energy savings required to switch
ACCURACY_DROP_TOLERANCE = 0.02  # Max 2% accuracy drop allowed

# Paths
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "green_metrics.sqlite3"
MODELS_PATH = ROOT / "models"
EMAIL_SAMPLES_PATH = ROOT / "data" / "email_samples.txt"

