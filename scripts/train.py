"""
Training script for LFW face recognition.
Run from project root: python scripts/train.py
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from src.data.lfw import create_lfw_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.classifier import FaceClassifier
from src.training.trainer import Trainer

def main():
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    LFW_ROOT = 'data'
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    MIN_IMAGES = 10
    MAX_PEOPLE = 50
    EPOCHS = 20
    LR = 0.001
    
    # Load data
    print('\nLoading LFW dataset...')
    train_loader, val_loader, num_classes, class_names = create_lfw_dataloaders(
        root=LFW_ROOT,
        batch_size=BATCH_SIZE,
        min_images_per_person=MIN_IMAGES,
        max_people=MAX_PEOPLE,
        transform_train=get_train_transforms(IMAGE_SIZE),
        transform_val=get_val_transforms(IMAGE_SIZE),
        num_workers=0  # Avoid multiprocessing issues on Windows
    )
    
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Number of people: {num_classes}')
    print(f'Sample people: {class_names[:5]}')
    
    # Create model
    print('\nCreating model...')
    model = FaceClassifier(
        num_classes=num_classes,
        backbone='resnet50',
        pretrained=True,
        dropout=0.3
    ).to(device)
    
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=True
    )
    
    # Train
    print('\nStarting training...')
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_dir='models/checkpoints'
    )
    
    # Save final model with class names
    import json
    save_dict = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
        'backbone': 'resnet50',
        'final_val_acc': history['val_acc'][-1]
    }
    torch.save(save_dict, 'models/checkpoints/face_recognition.pt')
    
    with open('models/checkpoints/class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f'\nTraining complete!')
    print(f'Best validation accuracy: {max(history["val_acc"]):.2f}%')
    print(f'Model saved to models/checkpoints/')

if __name__ == '__main__':
    main()
