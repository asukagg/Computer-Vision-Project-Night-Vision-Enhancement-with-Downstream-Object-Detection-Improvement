"""
Data Loading Module for Night Vision Enhancement Research
Handles LOL dataset (image enhancement) and ExDark dataset (object detection)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms

class LOLDataset(Dataset):
    """
    Low-Light (LOL) Dataset for Image Enhancement
    Structure: LOLdataset/eval15 or our485 -> {high, low}
    """
    
    def __init__(self, root_dir, split='train', image_size=400, augment=True):
        """
        Args:
            root_dir: Path to LOLdataset folder
            split: 'train', 'val', or 'test'
            image_size: Size to resize images to
            augment: Apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.split = split
        
        # Get all low-light and high-light image pairs
        self.low_images = []
        self.high_images = []
        
        # Check both eval15 and our485 folders
        for dataset_name in ['eval15', 'our485']:
            dataset_path = self.root_dir / dataset_name
            if dataset_path.exists():
                low_path = dataset_path / 'low'
                high_path = dataset_path / 'high'
                
                if low_path.exists() and high_path.exists():
                    low_files = sorted([f for f in os.listdir(low_path) 
                                       if f.endswith(('.jpg', '.png', '.jpeg'))])
                    high_files = sorted([f for f in os.listdir(high_path) 
                                        if f.endswith(('.jpg', '.png', '.jpeg'))])
                    
                    # Match pairs
                    for low_file in low_files:
                        base_name = os.path.splitext(low_file)[0]
                        # Try different extensions
                        for ext in ['.jpg', '.png', '.jpeg']:
                            high_file = base_name + ext
                            high_full_path = high_path / high_file
                            if high_full_path.exists():
                                self.low_images.append(low_path / low_file)
                                self.high_images.append(high_full_path)
                                break
        
        # Split data
        total_pairs = len(self.low_images)
        train_size = int(0.7 * total_pairs)
        val_size = int(0.15 * total_pairs)
        
        if split == 'train':
            indices = list(range(train_size))
        elif split == 'val':
            indices = list(range(train_size, train_size + val_size))
        else:  # test
            indices = list(range(train_size + val_size, total_pairs))
        
        self.low_images = [self.low_images[i] for i in indices]
        self.high_images = [self.high_images[i] for i in indices]
        
        print(f"LOL Dataset {split}: {len(self.low_images)} image pairs loaded")
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        # Read low and high images
        low_img = cv2.imread(str(self.low_images[idx]))
        high_img = cv2.imread(str(self.high_images[idx]))
        
        if low_img is None or high_img is None:
            return None
        
        # Convert BGR to RGB
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
        
        # Resize
        low_img = cv2.resize(low_img, (self.image_size, self.image_size))
        high_img = cv2.resize(high_img, (self.image_size, self.image_size))
        
        # Data augmentation
        if self.augment and self.split == 'train':
            # Random horizontal flip
            if random.random() > 0.5:
                low_img = cv2.flip(low_img, 1)
                high_img = cv2.flip(high_img, 1)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                h, w = low_img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                low_img = cv2.warpAffine(low_img, M, (w, h))
                high_img = cv2.warpAffine(high_img, M, (w, h))
        
        # Normalize to [0, 1]
        low_img = low_img.astype(np.float32) / 255.0
        high_img = high_img.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        low_tensor = torch.from_numpy(low_img.transpose(2, 0, 1))
        high_tensor = torch.from_numpy(high_img.transpose(2, 0, 1))
        
        return {
            'low': low_tensor,
            'high': high_tensor,
            'low_path': str(self.low_images[idx])
        }


class ExDarkDataset(Dataset):
    """
    ExDark Dataset for Object Detection
    Structure: ExDark/{class_name}/*.jpg
    """
    
    def __init__(self, root_dir, split='train', image_size=640, augment=True):
        """
        Args:
            root_dir: Path to ExDark folder
            split: 'train', 'val', or 'test'
            image_size: Size for detection
            augment: Apply augmentation
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.split = split
        
        # Class names (ExDark has 12 object categories)
        self.class_names = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
                           'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Collect all images with their class labels
        self.images = []
        self.labels = []
        
        for class_name in self.class_names:
            class_path = self.root_dir / class_name
            if class_path.exists():
                image_files = sorted([f for f in os.listdir(class_path)
                                     if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                for img_file in image_files:
                    self.images.append(class_path / img_file)
                    self.labels.append(self.class_to_idx[class_name])
        
        # Split data
        total_samples = len(self.images)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        if split == 'train':
            indices = indices[:train_size]
        elif split == 'val':
            indices = indices[train_size:train_size + val_size]
        else:  # test
            indices = indices[train_size + val_size:]
        
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        print(f"ExDark Dataset {split}: {len(self.images)} images loaded")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        label = self.labels[idx]
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Resize while maintaining aspect ratio
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to image_size
        top = (self.image_size - new_h) // 2
        left = (self.image_size - new_w) // 2
        img = cv2.copyMakeBorder(img, top, self.image_size - new_h - top,
                                 left, self.image_size - new_w - left,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Data augmentation
        if self.augment and self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random brightness/contrast
            if random.random() > 0.5:
                alpha = 1.0 + random.uniform(-0.3, 0.3)
                beta = random.uniform(-50, 50)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
        
        return {
            'image': img_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': img_path
        }


def get_lol_dataloader(root_dir, split='train', batch_size=16, 
                       image_size=400, num_workers=4):
    """Get LOL dataset dataloader"""
    dataset = LOLDataset(root_dir, split=split, image_size=image_size, 
                        augment=(split == 'train'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                           num_workers=num_workers, pin_memory=True)
    return dataloader


def get_exdark_dataloader(root_dir, split='train', batch_size=16, 
                         image_size=640, num_workers=4):
    """Get ExDark dataset dataloader"""
    dataset = ExDarkDataset(root_dir, split=split, image_size=image_size,
                           augment=(split == 'train'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                           num_workers=num_workers, pin_memory=True)
    return dataloader


if __name__ == "__main__":
    # Test data loading
    print("Testing LOL Dataset:")
    lol_train = get_lol_dataloader("./LOLdataset", split='train', batch_size=4)
    batch = next(iter(lol_train))
    print(f"  Low batch shape: {batch['low'].shape}")
    print(f"  High batch shape: {batch['high'].shape}")
    
    print("\nTesting ExDark Dataset:")
    exdark_train = get_exdark_dataloader("./ExDark", split='train', batch_size=4)
    batch = next(iter(exdark_train))
    print(f"  Image batch shape: {batch['image'].shape}")
    print(f"  Label batch shape: {batch['label'].shape}")
