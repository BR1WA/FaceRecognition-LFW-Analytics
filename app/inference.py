"""
Face detection and recognition inference pipeline.
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image


class FaceRecognizer:
    """
    Complete pipeline for face detection + recognition.
    Uses YOLO for detection and trained classifier for recognition.
    """
    
    def __init__(
        self,
        classifier_path: str,
        class_names: List[str],
        yolo_model: str = "yolov8n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.5
    ):
        from ultralytics import YOLO
        from src.models.classifier import FaceClassifier
        from src.data.transforms import get_val_transforms
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.class_names = class_names
        
        # Load YOLO detector
        self.detector = YOLO(yolo_model)
        
        # Load classifier
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.classifier = FaceClassifier(
            num_classes=len(class_names),
            backbone="resnet50"
        )
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Transform for classifier input
        self.transform = get_val_transforms(224)
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using YOLO."""
        results = self.detector.predict(
            image, 
            classes=[0],  # Person class
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf
            })
        
        return detections
    
    def recognize_face(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """Classify a face crop."""
        # Convert to PIL and apply transforms
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.classifier(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        return self.class_names[pred_idx], confidence
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame: detect faces and recognize."""
        detections = self.detect_faces(frame)
        results = []
        
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            
            # Crop face with margin
            h, w = frame.shape[:2]
            margin = int(0.1 * (x2 - x1))
            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(w, x2 + margin)
            y2m = min(h, y2 + margin)
            
            face_crop = frame[y1m:y2m, x1m:x2m]
            
            if face_crop.size > 0:
                name, conf = self.recognize_face(face_crop)
                det["name"] = name
                det["recognition_conf"] = conf
                results.append(det)
                
                # Draw on frame
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, results
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> List[List[Dict]]:
        """Process entire video."""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        else:
            writer = None
        
        all_results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, results = self.process_frame(frame)
            all_results.append(results)
            
            if writer:
                writer.write(annotated)
        
        cap.release()
        if writer:
            writer.release()
        
        return all_results
