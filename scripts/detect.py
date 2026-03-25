from ultralytics import YOLO
import cv2
from collections import Counter
from scripts.config import VEHICLE_CLASSES, DEFAULT_MODEL_WEIGHTS

class VehicleDetector:
    def __init__(self, model_path=DEFAULT_MODEL_WEIGHTS):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
    def get_vehicle_detections(self, image, mode="multi_vehicle"):
        results = self.model(image, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            det = {
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
                "bbox": (x1, y1, x2, y2)
            }
            detections.append(det)
            
        if mode == "single_car":
            car_dets = [d for d in detections if d["cls_name"] == "car"]
            if not car_dets:
                return []
            best_car = max(car_dets, key=lambda d: d["conf"])
            return [best_car]
            
        elif mode == "multi_car":
            return [d for d in detections if d["cls_name"] == "car"]
            
        elif mode in ["multi_vehicle", "count_vehicles"]:
            return [d for d in detections if d["cls_name"] in VEHICLE_CLASSES]
            
        return detections

    def annotate_image(self, image, detections, mode="multi_vehicle"):
        annotated = image.copy()
        
        # Draw boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['cls_name']} {det['conf']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
        # Overlay counts if in counting mode
        if mode == "count_vehicles":
            counts = Counter(det["cls_name"] for det in detections)
            y0 = 30
            for cls_name, cnt in counts.items():
                text = f"{cls_name}: {cnt}"
                cv2.putText(annotated, text, (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y0 += 30
                
        return annotated

    def run_on_image(self, image_path, output_path, mode="multi_vehicle"):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        detections = self.get_vehicle_detections(image, mode=mode)
        annotated = self.annotate_image(image, detections, mode=mode)
        
        cv2.imwrite(output_path, annotated)
        print(f"Saved inference result to {output_path}")

    def run_on_video(self, video_path, output_path, mode="multi_vehicle", show=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = self.get_vehicle_detections(frame, mode=mode)
            annotated = self.annotate_image(frame, detections, mode=mode)
            
            out.write(annotated)
            
            if show:
                cv2.imshow("YOLO Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        out.release()
        print(f"Saved output video to {output_path}")
