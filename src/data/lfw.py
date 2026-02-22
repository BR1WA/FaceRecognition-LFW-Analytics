"""
LFW (Labeled Faces in the Wild) dataset loader for face recognition.

LFW structure:
    lfw-deepfunneled/
        lfw-deepfunneled/
            Aaron_Eckhart/
                Aaron_Eckhart_0001.jpg
            George_W_Bush/
                George_W_Bush_0001.jpg
                George_W_Bush_0002.jpg
                ...
"""
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
from collections import Counter
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd


class LFWDataset(Dataset):
    """
    LFW dataset for face recognition (identity classification).
    
    Args:
        root: Root directory containing LFW data
        split: 'train' or 'val' (80/20 split)
        transform: Image transformations
        min_images_per_person: Filter people with fewer images
        max_people: Maximum number of identities to use
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        min_images_per_person: int = 5,
        max_people: Optional[int] = None,
        train_ratio: float = 0.8
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.min_images = min_images_per_person
        self.train_ratio = train_ratio
        
        # Find image directory
        self.img_dir = self._find_img_dir()
        
        # Load and filter data
        self.samples, self.name_to_idx, self.idx_to_name = self._load_data(max_people)
        self.num_classes = len(self.name_to_idx)
        
        print(f"LFW {split}: {len(self.samples)} images, {self.num_classes} people")
    
    def _find_img_dir(self) -> Path:
        """Find the LFW image directory."""
        possible_paths = [
            self.root / "lfw-deepfunneled" / "lfw-deepfunneled",
            self.root / "lfw-deepfunneled",
            self.root / "lfw",
            self.root,
        ]
        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                # Check if contains person folders
                for item in path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        return path
        raise FileNotFoundError(f"Could not find LFW images in {self.root}")
    
    def _load_data(self, max_people: Optional[int]) -> Tuple[List, Dict, Dict]:
        """Load LFW data with filtering."""
        # Collect all person folders and their images
        person_images = {}
        for person_dir in self.img_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith('.'):
                continue
            
            images = list(person_dir.glob("*.jpg"))
            if len(images) >= self.min_images:
                person_images[person_dir.name] = images
        
        # Sort by number of images (most first) and limit
        sorted_people = sorted(person_images.keys(), 
                               key=lambda x: len(person_images[x]), 
                               reverse=True)
        
        if max_people:
            sorted_people = sorted_people[:max_people]
        
        # Create name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(sorted(sorted_people))}
        idx_to_name = {idx: name for name, idx in name_to_idx.items()}
        
        # Split into train/val
        all_samples = []
        for name in sorted_people:
            images = person_images[name]
            for img_path in images:
                all_samples.append((img_path, name_to_idx[name], name))
        
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(all_samples)
        
        # Split
        split_idx = int(len(all_samples) * self.train_ratio)
        if self.split == "train":
            samples = all_samples[:split_idx]
        else:
            samples = all_samples[split_idx:]
        
        # Convert to (path, label) format
        samples = [(s[0], s[1]) for s in samples]
        
        return samples, name_to_idx, idx_to_name
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_person_name(self, label: int) -> str:
        """Get person name from class label."""
        return self.idx_to_name.get(label, "Unknown")
    
    def get_class_names(self) -> List[str]:
        """Get list of all person names."""
        return [self.idx_to_name[i] for i in range(self.num_classes)]


def create_lfw_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    min_images_per_person: int = 10,
    max_people: int = 50,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """
    Create LFW dataloaders for training.
    
    Args:
        root: Path to LFW data directory
        batch_size: Batch size
        num_workers: Data loading workers
        min_images_per_person: Filter people with fewer images
        max_people: Maximum number of people to include
        transform_train: Training transforms
        transform_val: Validation transforms
    
    Returns:
        train_loader, val_loader, num_classes, class_names
    """
    train_dataset = LFWDataset(
        root=root,
        split="train",
        transform=transform_train,
        min_images_per_person=min_images_per_person,
        max_people=max_people
    )
    
    val_dataset = LFWDataset(
        root=root,
        split="val",
        transform=transform_val,
        min_images_per_person=min_images_per_person,
        max_people=max_people
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return (train_loader, val_loader, 
            train_dataset.num_classes, train_dataset.get_class_names())
