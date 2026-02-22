
"""Video Face Recognition Pipeline using YOLO + Trained Classifier."""

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

from src.models.classifier import FaceClassifier
from src.data.transforms import get_val_transforms


class VideoFaceRecognizer:
    """Pipeline for detecting and recognizing faces in videos."""

    def __init__(self, face_detector, face_recognizer, class_names, transform, 
                 device='cuda', confidence_threshold=0.5):
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.class_names = class_names
        self.transform = transform
        self.device = device
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_checkpoint(cls, checkpoint_path, yolo_model='yolov8n.pt', device=None):
        """Create VideoFaceRecognizer from saved checkpoint."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load face detector
        face_detector = YOLO(yolo_model)

        # Load face recognizer
        checkpoint = torch.load(checkpoint_path, map_location=device)

        num_classes = checkpoint['num_classes']
        class_names = checkpoint['class_names']
        backbone = checkpoint.get('backbone', 'resnet50')

        face_recognizer = FaceClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,
            dropout=0.3
        ).to(device)

        face_recognizer.load_state_dict(checkpoint['model_state_dict'])
        face_recognizer.eval()

        # Transform
        transform = get_val_transforms(224)

        return cls(face_detector, face_recognizer, class_names, transform, device)

    def detect_faces(self, frame):
        """Detect faces in a frame using YOLO."""
        results = self.face_detector(frame, verbose=False)
        boxes = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((x1, y1, x2, y2, conf))

        return boxes

    def crop_face_region(self, frame, box, expand_ratio=0.2):
        """Crop and preprocess face region from frame."""
        x1, y1, x2, y2, _ = box
        h, w = frame.shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1

        x1 = max(0, int(x1 - box_w * expand_ratio))
        y1 = max(0, int(y1 - box_h * expand_ratio))
        x2 = min(w, int(x2 + box_w * expand_ratio))
        y2 = min(h, int(y2 + box_h * expand_ratio))

        face_crop = frame[y1:y2, x1:x2]
        return face_crop, (x1, y1, x2, y2)

    def recognize_face(self, face_crop):
        """Recognize a face using the trained model."""
        if face_crop.size == 0:
            return None, 0.0

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.face_recognizer(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(dim=1)

        pred_idx = pred.item()
        confidence = conf.item()
        name = self.class_names[pred_idx]

        return name, confidence

    def process_frame(self, frame, recognition_threshold=0.3):
        """Process a single frame: detect faces and recognize them."""
        detections = self.detect_faces(frame)
        results = []

        for box in detections:
            face_crop, expanded_box = self.crop_face_region(frame, box)
            name, confidence = self.recognize_face(face_crop)

            if name and confidence >= recognition_threshold:
                results.append({'box': expanded_box, 'name': name, 'confidence': confidence})
            else:
                results.append({'box': expanded_box, 'name': 'Unknown', 'confidence': confidence or 0.0})

        return results

    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        frame_draw = frame.copy()

        for result in results:
            x1, y1, x2, y2 = result['box']
            name = result['name'].replace('_', ' ')
            conf = result['confidence']

            color = (0, 0, 255) if result['name'] == 'Unknown' else (0, 255, 0)

            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)

            label = f'{name} ({conf:.2f})'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_draw, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame_draw

    def process_video(self, input_path, output_path, recognition_threshold=0.3, show_progress=True):
        """Process entire video and save with face recognition annotations."""
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {input_path}')

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        pbar = tqdm(total=total_frames, desc='Processing video') if show_progress else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.process_frame(frame, recognition_threshold)
            annotated_frame = self.draw_results(frame, results)
            out.write(annotated_frame)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        cap.release()
        out.release()

        return output_path
