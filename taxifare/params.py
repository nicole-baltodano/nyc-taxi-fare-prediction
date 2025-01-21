import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "1k" # ["1k", "200k", "all"]
CHUNK_SIZE = 200
GCP_PROJECT = "<your project id>" # TO COMPLETE
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"
##################  CONSTANTS  #####################
# Path to the 'data' folder one level above the current working directory
LOCAL_DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))

# Path to the 'registry' folder in the current working directory
LOCAL_REGISTRY_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "training_outputs"))

COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

DTYPES_RAW = {
    "fare_amount": "float32",
    "pickup_datetime": "datetime64[ns, UTC]",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16"
}

DTYPES_PROCESSED = np.float32

