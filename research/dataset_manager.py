import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DeepfakeDataset(ABC):
    def __init__(self, name, data_path, license_accepted=False):
        self.name = name
        self.data_path = data_path
        self.license_accepted = license_accepted
        
    @abstractmethod
    def verify_structure(self):
        """Verifies if the required folder structure exists."""
        pass

    @abstractmethod
    def get_dataloaders(self, batch_size=32, num_workers=4):
        """Returns train and validation PyTorch dataloaders."""
        pass

class DatasetManager:
    """Centralized manager for research datasets."""
    def __init__(self, base_data_dir="./data"):
        self.base_data_dir = base_data_dir
        self.datasets = {}
        
    def register_dataset(self, dataset: DeepfakeDataset):
        if not dataset.license_accepted:
            logger.warning(f"Registration failed for {dataset.name}. Explicit license acceptance required.")
            return False
            
        if not dataset.verify_structure():
            logger.warning(f"Registration failed for {dataset.name}. Invalid structure at {dataset.data_path}")
            return False
            
        self.datasets[dataset.name] = dataset
        logger.info(f"Registered dataset: {dataset.name}")
        return True

    def get_loader(self, dataset_name, split="train", **kwargs):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not registered or downloaded.")
            
        loaders = self.datasets[dataset_name].get_dataloaders(**kwargs)
        if split == "train":
            return loaders[0]
        elif split == "val":
            return loaders[1]
        else:
            raise ValueError(f"Unknown split: {split}")

# --- Example implementation for FaceForensics++ (ff-c23) ---
class FaceForensicsC23(DeepfakeDataset):
    def __init__(self, base_dir, license_accepted=False):
        path = os.path.join(base_dir, "ff-c23")
        super().__init__("FaceForensics++ C23", path, license_accepted)
        
    def verify_structure(self):
        # Simply checks if train and val exist inside ff-c23
        return os.path.exists(os.path.join(self.data_path, "train")) and \
               os.path.exists(os.path.join(self.data_path, "val"))
               
    def get_dataloaders(self, batch_size=32, num_workers=4):
        # We would import torch and return the DataLoaders here like in train_efficientnet.py
        # For the stub, we return None
        logger.info(f"Loading FaceForensics++ C23 from {self.data_path}")
        return None, None
