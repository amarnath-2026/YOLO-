import os

# Define constants
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "bicycle"]
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
