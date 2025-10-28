import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "data", "raw", "alzheimers_disease_data.csv")
)
PROCESSED_DATA_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
)
BEST_MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "model training", "output", "trained_models", "best_model.pkl")
)

# Default directory for prediction outputs (CSV results from predict.py)
PREDICTIONS_DIR = os.path.normpath(os.path.join(BASE_DIR, "prediction"))
DEFAULT_BATCH_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "test_eval_results.csv")

