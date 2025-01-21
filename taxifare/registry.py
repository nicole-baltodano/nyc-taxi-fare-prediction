import glob
import os
import time
import pickle
from pathlib import Path
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from params import *

# Set the local registry path to the directory of the current Python file
LOCAL_REGISTRY_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at:
    "{LOCAL_REGISTRY_PATH}/params/params_{timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/metrics_{timestamp}.pickle"
    """

    # Generate a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Define directories for params and metrics
    params_dir = LOCAL_REGISTRY_PATH / "params"
    metrics_dir = LOCAL_REGISTRY_PATH / "metrics"

    # Ensure directories exist
    params_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save params locally
    if params is not None:
        params_path = params_dir / f"params_{timestamp}.pickle"
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
        print(f"✅ Params saved at {params_path}")

    # Save metrics locally
    if metrics is not None:
        metrics_path = metrics_dir / f"metrics_{timestamp}.pickle"
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)
        print(f"✅ Metrics saved at {metrics_path}")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at:
    "{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    # Generate a timestamp for the file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Define the model directory and ensure it exists
    model_directory = LOCAL_REGISTRY_PATH / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    # Define the full path for the model file
    model_path = model_directory / f"model_{timestamp}.h5"

    # Save the model
    model.save(model_path)

    print(f"✅ Model saved locally at {model_path}")


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    """

    MODEL_TARGET = 'local'

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Define the model directory and list available models
        model_directory = LOCAL_REGISTRY_PATH / "models"
        local_model_paths = list(model_directory.glob("*.h5"))

        if not local_model_paths:
            print(Fore.RED + "❌ No models found in the local registry." + Style.RESET_ALL)
            return None

        # Get the most recent model file
        most_recent_model_path = sorted(local_model_paths)[-1]
        print(Fore.BLUE + f"\nLoading model from: {most_recent_model_path}" + Style.RESET_ALL)

        # Load and return the model
        latest_model = keras.models.load_model(most_recent_model_path)
        print(Fore.GREEN + "✅ Model loaded successfully from local disk" + Style.RESET_ALL)

        return latest_model

