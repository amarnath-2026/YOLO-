import argparse
import os
from scripts.detect import VehicleDetector
from scripts.train import download_roboflow_dataset, train_model
from scripts.config import OUTPUTS_DIR

def main():
    parser = argparse.ArgumentParser(description="YOLO Vehicle Detection and Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Detection parsing
    detect_parser = subparsers.add_parser("detect", help="Run detection on image or video")
    detect_parser.add_argument("--source", type=str, required=True, help="Path to input image or video")
    detect_parser.add_argument("--mode", type=str, default="multi_vehicle", choices=["single_car", "multi_car", "multi_vehicle", "count_vehicles"], help="Detection mode")
    detect_parser.add_argument("--type", type=str, choices=["image", "video"], required=True, help="Type of the input source")
    detect_parser.add_argument("--output_name", type=str, required=True, help="Filename of the output (e.g., result.jpg or result.mp4)")

    # Training parsing
    train_parser = subparsers.add_parser("train", help="Download dataset and fine-tune YOLO model")
    train_parser.add_argument("--api_key", type=str, help="Roboflow API key", required=True)
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    train_parser.add_argument("--workspace", type=str, default="roboflow-gw7yv", help="Roboflow workspace")
    train_parser.add_argument("--project", type=str, default="vehicles-openimages", help="Roboflow project name")

    args = parser.parse_args()

    if args.command == "detect":
        detector = VehicleDetector()
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUTS_DIR, args.output_name)
        
        if args.type == "image":
            detector.run_on_image(args.source, out_path, mode=args.mode)
        elif args.type == "video":
            detector.run_on_video(args.source, out_path, mode=args.mode)

    elif args.command == "train":
        print("Downloading dataset from Roboflow...")
        dataset_path = download_roboflow_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project_name=args.project,
            version_num=1
        )
        data_yaml = os.path.join(dataset_path, "data.yaml")
        print(f"Starting YOLO model training using dataset config mapping to: {data_yaml}")
        train_model(data_yaml, epochs=args.epochs)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
