import os
from roboflow import Roboflow
from ultralytics import YOLO
from scripts.config import DEFAULT_MODEL_WEIGHTS, DATA_DIR

def download_roboflow_dataset(api_key, workspace, project_name, version_num):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    
    # Store the current path, switch to data dir for download
    origin_cwd = os.getcwd()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(DATA_DIR)
    
    dataset = version.download("yolov8")
    
    os.chdir(origin_cwd)
    return os.path.join(DATA_DIR, dataset.location)

def train_model(data_yaml_path, epochs=5, imgsz=640, batch=16, lr0=0.0001, model_weights=DEFAULT_MODEL_WEIGHTS):
    model = YOLO(model_weights)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0
    )
    print("Training process finished.")
