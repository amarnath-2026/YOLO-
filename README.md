# YOLO Object Detection and Fine-Tuning

This repository contains an object detection pipeline utilizing the `ultralytics` YOLOv8 model and `roboflow`. The project supports single and multi-vehicle detection, vehicle counting, and fine-tuning on custom datasets.

## Project Structure

- `notebooks/YOLO.ipynb`: Main Jupyter Notebook containing the code for loading models, inference on images/videos, and training/fine-tuning pipelines.
- `data/`: Directory to store datasets. (Ignored in Git)
- `models/`: Directory to store downloaded YOLO weights (e.g., `yolov8n.pt`). (Ignored in Git)
- `outputs/`: Directory to store annotated images and videos. (Ignored in Git)
- `scripts/`: Directory for any standalone python scripts extracted.
- `requirements.txt`: Required dependencies.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd YOLO
   ```

2. **Install dependencies:**
   It is recommended to use a virtual Python environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   Launch Jupyter Notebook or Jupyter Lab and open `notebooks/YOLO.ipynb`.
   ```bash
   jupyter notebook notebooks/YOLO.ipynb
   ```

## Push to GitHub
If you are the owner and need to push this to your own GitHub remote:
```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
