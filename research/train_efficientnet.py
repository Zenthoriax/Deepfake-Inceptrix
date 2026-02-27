import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from tqdm import tqdm
import time

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Assumes dataset downloaded from Kaggle: https://www.kaggle.com/datasets/xdxd003/ff-c23
# Use: kaggle datasets download -d xdxd003/ff-c23 && unzip ff-c23.zip -d ./data
DATA_DIR = "./data/ff-c23" # Update to the extracted kaggle path
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 2 # Fake vs Real
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_transforms():
    """Returns data augmentation and standard transforms for EfficientNet."""
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)), # EfficientNet-B4 expected size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_data():
    """Loads datasets using ImageFolder assuming a structure like train/real, train/fake."""
    train_tfms, val_tfms = get_transforms()
    
    if not os.path.exists(TRAIN_DIR):
        print(f"Warning: Training directory {TRAIN_DIR} not found. Please download the dataset.")
        return None, None
        
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_tfms)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_tfms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def build_model():
    """Loads pretrained EfficientNet-B4 and replaces the classifier head."""
    print("Loading pretrained EfficientNet-B4...")
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    
    # Freeze early layers for fine-tuning
    for param in model.features[:-3].parameters():
        param.requires_grad = False
        
    # Replace the classification head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    return model.to(DEVICE)

def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # MPS does not support half precision for all operations, so conditionally use autocast
            with torch.amp.autocast(device_type=DEVICE.type) if DEVICE.type == 'cuda' else torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = load_data()
    
    if not train_loader:
        print("Skipping training as data is not present.")
        print("Please download using: kaggle datasets download -d xdxd003/ff-c23")
        return
        
    model = build_model()
    
    # Class weights if dataset is imbalanced (optional)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW for better weight decay handling
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Setup mixed precision scaler
    # Note: GradScaler is for CUDA. Disable for MPS/CPU.
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Time: {time.time() - start_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("=> Saving new best model!")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, 'checkpoints/efficientnet_b4_ff_c23_best.pth')

if __name__ == '__main__':
    main()
