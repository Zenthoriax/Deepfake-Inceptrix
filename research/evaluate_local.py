import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import os
import json
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_DIR = "train" 
BATCH_SIZE = 64 # Larger batch for faster eval
NUM_CLASSES = 2 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_transforms():
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return val_transform

def build_model():
    print("Loading pretrained EfficientNet-B4...")
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    # Load the proxy-trained weights if available, or just evaluate the untrained architecture baseline
    chkpt_path = 'checkpoints/efficientnet_b4_streaming_chkpt.pth'
    if os.path.exists(chkpt_path):
        print(f"Loading weights from {chkpt_path}...")
        checkpoint = torch.load(chkpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No trained checkpoints found. Evaluating raw pre-trained architecture baseline.")
        
    return model.to(DEVICE)

def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating on Celeb-DF"):
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
        
    eval_time = time.time() - start_time
    return final_loss, acc, prec, rec, f1, auc, eval_time, len(all_targets)

def run_evaluation():
    print("Preparing Local Celeb-DF Validation Set...")
    val_tfm = get_transforms()
    
    # Load same full dataset, then extract the 20% validation split (seed fixed for consistency)
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfm)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Executing Full Matrix Evaluation on {val_size} validation images...")
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    
    val_loss, acc, prec, rec, f1, auc, eval_time, total_samples = evaluate_model(model, val_loader, criterion)
    
    print(f"\nEvaluation Complete! Time: {eval_time:.1f}s")
    print(f"Loss: {val_loss:.4f} | Acc: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    # Generate JSON Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "Local Celeb-DF Validation",
        "dataset_path": DATA_DIR,
        "total_evaluation_samples": total_samples,
        "evaluation_time_seconds": round(eval_time, 2),
        "metrics": {
            "loss": round(val_loss, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4)
        },
        "model_architecture": "EfficientNet-B4"
    }
    
    os.makedirs('reports', exist_ok=True)
    report_path = f"reports/celeb_df_eval_report_{int(time.time())}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"=> Evaluation report saved to {report_path}")

if __name__ == '__main__':
    run_evaluation()
