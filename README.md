# Face Recognition with LFW

A deep learning project for face detection and recognition using PyTorch.

## Features
- Face detection using YOLOv8
- Face recognition (identify people by name)
- Trained on LFW (Labeled Faces in the Wild) dataset
- Video inference pipeline

## Project Structure
```
├── app/                    # Inference pipeline
│   └── inference.py        # FaceRecognizer class
├── data/                   # LFW dataset
├── models/checkpoints/     # Saved model weights
├── notebooks/
│   └── train_lfw.ipynb     # Training notebook
├── outputs/                # Training outputs
├── src/
│   ├── data/
│   │   ├── lfw.py          # LFW dataset loader
│   │   └── transforms.py   # Data augmentation
│   ├── models/
│   │   └── classifier.py   # Face classifier
│   └── training/
│       └── trainer.py      # Training loop
└── requirements.txt
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Open `notebooks/train_lfw.ipynb` and run training

3. Use trained model for inference:
```python
from app.inference import FaceRecognizer

recognizer = FaceRecognizer(
    classifier_path='models/checkpoints/best_model.pt',
    class_names=['George_W_Bush', 'Colin_Powell', ...]
)

# Process video
recognizer.process_video('input.mp4', 'output.mp4')
```

## Dataset

Using LFW (Labeled Faces in the Wild):
- 13,000+ face images
- 5,749 people
- Images organized by person name
