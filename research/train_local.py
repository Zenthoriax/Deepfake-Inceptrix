import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
 
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_DIR = "train" # Output dir of extract_frames.py
BATCH_SIZE = 32
NUM_CLASSES = 2 # Fake vs Real
LEARNING_RATE = 1e-4
EPOCHS = 10 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_transforms():
    """Returns data augmentation for EfficientNet."""
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
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

def build_model():
    """Loads pretrained EfficientNet-B4 and replaces the classifier head."""
    print("Loading pretrained EfficientNet-B4...")
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    
    # Freeze early layers for fine-tuning speed
    for param in model.features[:-3].parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    return model.to(DEVICE)

def evaluate_model(model, val_loader, criterion):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast(device_type=DEVICE.type) if DEVICE.type == 'cuda' else torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    final_loss = running_loss / len(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5
        
    return final_loss, acc, prec, rec, f1, auc

def train_local(epochs=EPOCHS):
    print("Preparing Local ImageFolder Dataset...")
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: Could not find training data in {DATA_DIR}.")
        print("Please run `python research/extract_frames.py` first to generate the dataset from DATA/ videos.")
        return

    train_tfm, val_tfm = get_transforms()
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
    
    # Check class to index mapping
    print(f"Class Mapping: {full_dataset.class_to_idx}")
    
    # Split 80/20 train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Override validation transform
    val_dataset.dataset.transform = val_tfm
    
    print(f"Training on {train_size} images. Validating on {val_size} images.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"Starting local training on {DEVICE} for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            if DEVICE.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
        # Validation Phase
        val_loss, acc, prec, rec, f1, auc = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | Val Acc: {acc*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    print(f"\nTraining Complete! Total Time: {time.time() - start_time:.1f}s")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoints/efficientnet_b4_local_chkpt.pth')
    print("=> Checkpoint saved to checkpoints/efficientnet_b4_local_chkpt.pth")
    
    # Generate JSON Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "training_type": "Local ImageFolder",
        "dataset_path": DATA_DIR,
        "total_train_samples": train_size,
        "total_val_samples": val_size,
        "epochs": epochs,
        "metrics": {
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(acc, 4),
            "val_precision": round(prec, 4),
            "val_recall": round(rec, 4),
            "val_f1_score": round(f1, 4),
            "val_auc_roc": round(auc, 4)
        }
    }
    
    os.makedirs('reports', exist_ok=True)
    report_path = f"reports/local_training_report_{int(time.time())}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"=> Final evaluation report saved to {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    args = parser.parse_args()
    
    train_local(epochs=args.epochs)
