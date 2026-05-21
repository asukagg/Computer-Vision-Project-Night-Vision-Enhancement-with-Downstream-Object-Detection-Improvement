"""
ExDark Object Classifier
Train CNN classifier on 12 ExDark object classes
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

class ExDarkDataset(Dataset):
    """Load ExDark dataset for classification"""
    
    def __init__(self, root_dir, image_size=224, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        self.images = []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append((os.path.join(cls_path, img_file), cls_idx))
        
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
        
        img = img.astype(np.float32) / 255.0
        img = transforms.functional.to_tensor(img)
        
        return img, label


class ObjectClassifier(nn.Module):
    """Simple CNN for 12-class object classification"""
    
    def __init__(self, num_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_classifier(data_dir, epochs=20, batch_size=32, lr=0.001):
    """Train ExDark classifier"""
    
    print("=" * 80)
    print("EXDARK OBJECT CLASSIFIER TRAINING")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Create dataset
    print("\nLoading ExDark dataset...")
    dataset = ExDarkDataset(data_dir, image_size=224)
    print(f"✓ Dataset loaded: {len(dataset)} images")
    print(f"✓ Classes ({len(dataset.classes)}): {', '.join(dataset.classes)}")
    
    # Create dataloader (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = ObjectClassifier(num_classes=len(dataset.classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created - {total_params:,} parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_acc = 0
    patience_count = 0
    
    print("\n" + "=" * 80)
    print("Training Started")
    print("=" * 80 + "\n")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
        
        train_acc = 100 * train_correct / len(train_dataset)
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        val_acc = 100 * val_correct / len(val_dataset)
        val_loss /= len(val_loader)
        
        print(f"✓ Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(), 'checkpoints/exdark_classifier.pth')
            print(f"  ✓ Best checkpoint saved (val_acc: {val_acc:.2f}%)")
        else:
            patience_count += 1
        
        scheduler.step(val_loss)
        
        if patience_count >= 5:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETED")
    print(f"✓ Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"✓ Model saved to: checkpoints/exdark_classifier.pth")
    print("=" * 80)
    
    return model, dataset


if __name__ == "__main__":
    train_classifier("ExDark", epochs=20, batch_size=32, lr=0.001)
