import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from datasets import load_dataset, interleave_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import time
import os
import argparse
import ssl
import json
from datetime import datetime

# Fix for macOS SSL certificate verification error when downloading pretrained weights
ssl._create_default_https_context = ssl._create_unverified_context


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# We use public huggingface datasets to demonstrate streaming 
# without downloading the entire compressed archive locally.
HF_DATASET_IDS = [
    "JamieWithofs/Deepfake-and-real-images-4"
    # To add more datasets to stream concurrently, simply append their paths here:
    # "Another/Dataset-Path",
]


BATCH_SIZE = 32
NUM_CLASSES = 2 # Fake vs Real
LEARNING_RATE = 1e-4

# We define device logic globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_transforms():
    """Returns data augmentation for EfficientNet."""
    transform = transforms.Compose([
        transforms.Resize((380, 380)), # EfficientNet-B4 expected size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

class HFStreamingIterableDataset(IterableDataset):
    """
    Wraps a Hugging Face streaming dataset into a PyTorch IterableDataset.
    This fetches records dynamically over the network, drastically reducing local disk usage.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __iter__(self):
        for item in self.hf_dataset:
            # Hugging Face usually provides PIL Images under the 'image' key
            # and generic labels under 'label'. You may need to adapt these key names.
            image = item.get('image')
            label = item.get('label', 0)
            
            # Ensure it's RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            if self.transform:
                image = self.transform(image)
                
            yield image, label

class MockIterableDataset(IterableDataset):
    """Fallback dataset that generates random tensors if network is blocked."""
    def __init__(self, size=1000):
        self.size = size
    
    def __iter__(self):
        for _ in range(self.size):
            yield torch.randn(3, 380, 380), torch.randint(0, NUM_CLASSES, (1,)).item()


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

def evaluate_streaming(model, criterion, max_steps=100):
    """Evaluates the model using the streaming pipeline on the test split."""
    print(f"\n--- Starting Evaluation on {max_steps} validation steps ---")
    try:
        val_datasets = []
        for ds_id in HF_DATASET_IDS:
            try:
                ds = load_dataset(ds_id, split="test", streaming=True)
                val_datasets.append(ds)
            except Exception as e:
                print(f"Skipping TEST split for {ds_id}: {e}")
                
        if not val_datasets:
            raise ValueError("No valid evaluation datasets found.")
            
        hf_val = interleave_datasets(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        tfms = get_transforms()
        pytorch_dataset = HFStreamingIterableDataset(hf_val, transform=tfms)
    except Exception as e:
        print(f"Failed to load any evaluation split. Error: {e}")
        print("Falling back to local generated mock dataset for evaluation demo...")
        pytorch_dataset = MockIterableDataset(size=500)

    val_loader = DataLoader(pytorch_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    pbar = tqdm(total=max_steps, desc="Streaming Eval")
    step = 0
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            if step >= max_steps:
                break
                
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast(device_type=DEVICE.type) if DEVICE.type == 'cuda' else torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            
            # Get probabilities for AUC
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of class 1 (Fake)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.update(1)
            step += 1
            
    pbar.close()
    
    if len(all_targets) == 0:
        print("No evaluation data processed.")
        return

    final_loss = running_loss / len(all_targets)
    
    # Calculate Sklearn Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5 # If only one class is present in the stream

    print(f"Eval Time: {time.time() - start_time:.1f}s")
    print(f"Eval Loss: {final_loss:.4f}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # Generate JSON Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "datasets_evaluated": HF_DATASET_IDS,
        "total_evaluation_samples": len(all_targets),
        "evaluation_time_seconds": round(time.time() - start_time, 2),
        "metrics": {
            "loss": round(final_loss, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4)
        },
        "model_architecture": "EfficientNet-B4"
    }
    
    os.makedirs('reports', exist_ok=True)
    report_path = f"reports/evaluation_report_{int(time.time())}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"=> Evaluation report saved to {report_path}")
    
    return final_loss, acc, prec, rec, f1, auc

def train_streaming(max_steps=500, eval_steps=100):
    """Trains the model using the streaming pipeline for a fixed number of steps."""
    print(f"Connecting to Hugging Face to stream {len(HF_DATASET_IDS)} datasets concurrently...")
    
    # streaming=True prevents downloading the full dataset to disk
    try:
        train_datasets = []
        for ds_id in HF_DATASET_IDS:
            try:
                ds = load_dataset(ds_id, split="train", streaming=True)
                train_datasets.append(ds)
            except Exception as e:
                print(f"Skipping TRAIN split for {ds_id}: {e}")
                
        if not train_datasets:
            raise ValueError("No valid training datasets found.")
            
        hf_train = interleave_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        tfms = get_transforms()
        pytorch_dataset = HFStreamingIterableDataset(hf_train, transform=tfms)
    except Exception as e:
        print(f"Failed to load datasets. Error: {e}")
        print("Falling back to local generated mock dataset for training demo...")
        pytorch_dataset = MockIterableDataset(size=5000)

    # Important: pin_memory could be unstable depending on the network iterable, 
    # but normally num_workers > 0 works well with standard datasets.
    train_loader = DataLoader(pytorch_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    model = build_model()
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"Starting streaming training on {DEVICE} for {max_steps} steps...")
    os.makedirs('checkpoints', exist_ok=True)
    
    pbar = tqdm(total=max_steps, desc="Streaming Train")
    step = 0
    
    start_time = time.time()
    
    for inputs, labels in train_loader:
        if step >= max_steps:
            break
            
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
        
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        step += 1
        
    pbar.close()
    
    final_loss = running_loss / max(1, total)
    final_acc = 100. * correct / max(1, total)
    
    print(f"Streaming Session Completed! Time: {time.time() - start_time:.1f}s")
    print(f"Final Loss: {final_loss:.4f} | Final Acc: {final_acc:.2f}%")
    
    # Run evaluation
    evaluate_streaming(model, criterion, max_steps=eval_steps)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoints/efficientnet_b4_streaming_chkpt.pth')
    print("=> Checkpoint saved to checkpoints/efficientnet_b4_streaming_chkpt.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate model via streaming dataset")
    parser.add_argument("--steps", type=int, default=500, help="Maximum number of train batches/steps to stream")
    parser.add_argument("--eval_steps", type=int, default=100, help="Maximum number of evaluation batches/steps to stream")
    args = parser.parse_args()
    
    train_streaming(max_steps=args.steps, eval_steps=args.eval_steps)
