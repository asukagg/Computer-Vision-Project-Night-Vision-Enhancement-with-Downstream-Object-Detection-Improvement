"""
Improved Lightweight Model - Better Architecture
Enhanced CNN with better feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedEnhancer(nn.Module):
    """Improved enhancement model with better architecture"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with batch norm
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck with residual
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128)
        )
        
        # Decoder with batch norm
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.final = nn.Sequential(
            nn.Conv2d(32 + 32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Initial
        initial = self.initial(x)
        
        # Encoder
        enc1 = self.enc1(initial)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        
        # Bottleneck with residual
        bottleneck = self.bottleneck(enc3)
        bottleneck = bottleneck + enc3
        
        # Decoder
        dec1 = self.dec1(bottleneck)
        dec1 = torch.cat([dec1, enc2], dim=1)
        
        dec2 = self.dec2(dec1)
        dec2 = torch.cat([dec2, enc1], dim=1)
        
        dec3 = self.dec3(dec2)
        dec3 = torch.cat([dec3, initial], dim=1)
        
        # Output with residual
        output = self.final(dec3)
        output = output + x * 0.3  # Stronger residual connection
        
        return torch.clamp(output, 0, 1)


class ImprovedDetector(nn.Module):
    """Lightweight detector"""
    
    def __init__(self, num_classes=12):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


class ImprovedDomainAdapter(nn.Module):
    """Improved domain adaptation module"""
    
    def __init__(self, feature_dim=128, num_classes=12):
        super().__init__()
        
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, feature_dim)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, features):
        adapted = self.feature_adapter(features)
        domain_logits = self.domain_classifier(features)
        return adapted, domain_logits
