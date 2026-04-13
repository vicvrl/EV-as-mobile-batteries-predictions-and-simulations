import os

DATA_FOLDER = "data_ev/"
TRIPS_PATH = os.path.join(DATA_FOLDER, "trips_data")
METRICS_PATH = "metrics"

EV_PATHS = [f"{DATA_FOLDER}/{file}" for file in os.listdir(DATA_FOLDER) if file.startswith("EV")]

RESULT_DIR = "results"

