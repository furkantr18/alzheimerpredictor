import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "data", "raw", "alzheimers_disease_data.csv")
)

