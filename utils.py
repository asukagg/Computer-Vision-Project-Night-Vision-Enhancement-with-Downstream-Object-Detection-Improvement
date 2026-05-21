"""
Utility Functions and Helpers
Image metrics, device management, and helper functions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import os
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


class MetricsCalculator:
    """Calculate image quality and detection metrics"""
    
    @staticmethod
    def calculate_psnr(img1, img2, max_val=1.0):
        """
        Calculate Peak Signal-to-Noise Ratio
        Args:
            img1, img2: Images in range [0, 1] or [0, 255]
            max_val: Maximum pixel value
        Returns:
            PSNR value in dB
        """
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # Ensure images are in range [0, 1]
        if img1.max() > 1.5:
            img1 = img1 / 255.0
        if img2.max() > 1.5:
            img2 = img2 / 255.0
        
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.0
        
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def calculate_ssim(img1, img2, max_val=1.0):
        """
        Calculate Structural Similarity Index
        Args:
            img1, img2: Images in range [0, 1] or [0, 255]
            max_val: Maximum pixel value
        Returns:
            SSIM value in range [-1, 1]
        """
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # Convert to [0, 1] if needed
        if img1.max() > 1.5:
            img1 = img1 / 255.0
        if img2.max() > 1.5:
            img2 = img2 / 255.0
        
        # Handle different image formats
        if len(img1.shape) == 3:
            if img1.shape[0] == 3:  # (C, H, W)
                img1 = np.transpose(img1, (1, 2, 0))
            if img1.shape[0] == 1:  # (1, H, W)
                img1 = img1[0]
        
        if len(img2.shape) == 3:
            if img2.shape[0] == 3:
                img2 = np.transpose(img2, (1, 2, 0))
            if img2.shape[0] == 1:
                img2 = img2[0]
        
        ssim = compute_ssim(img1, img2, data_range=max_val, channel_axis=None if len(img1.shape) == 2 else 2)
        return ssim
    
    @staticmethod
    def calculate_f1_score(tp, fp, fn):
        """
        Calculate F1 Score
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
        Returns:
            F1 score
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall


class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def normalize_image(img, mean=None, std=None):
        """Normalize image to [0, 1] or using given mean/std"""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        
        if img.max() > 1.5:
            img = img / 255.0
        
        if mean is not None and std is not None:
            img = F.normalize(img.unsqueeze(0), mean=mean, std=std).squeeze(0)
        
        return img
    
    @staticmethod
    def denormalize_image(img, mean=None, std=None):
        """Denormalize image"""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        
        if mean is not None and std is not None:
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
        
        img = torch.clamp(img, 0, 1)
        
        if isinstance(img, torch.Tensor):
            img = (img * 255).numpy().astype(np.uint8)
        
        return img
    
    @staticmethod
    def create_comparison_image(low_light, enhanced, reference, save_path=None):
        """
        Create side-by-side comparison image
        Args:
            low_light: Original low-light image
            enhanced: Enhanced image
            reference: Reference high-light image
            save_path: Path to save comparison image
        Returns:
            Comparison image
        """
        # Convert to numpy arrays if needed
        if isinstance(low_light, torch.Tensor):
            low_light = (low_light.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if isinstance(enhanced, torch.Tensor):
            enhanced = (enhanced.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if isinstance(reference, torch.Tensor):
            reference = (reference.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Ensure same size
        h, w = low_light.shape[:2]
        enhanced = cv2.resize(enhanced, (w, h))
        reference = cv2.resize(reference, (w, h))
        
        # Create comparison
        comparison = np.hstack([low_light, enhanced, reference])
        
        # Add labels
        cv2.putText(comparison, 'Low-Light', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(comparison, 'Enhanced', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Reference', (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, comparison)
        
        return comparison
    
    @staticmethod
    def apply_clahe(img, clip_limit=2.0, tile_size=8):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        if isinstance(img, torch.Tensor):
            img = (img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            result = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            result = clahe.apply(img)
        
        return result


class PathManager:
    """Manage output directories"""
    
    @staticmethod
    def setup_directories(base_dir, dirs=['checkpoints', 'logs', 'results', 'comparisons']):
        """Create necessary directories"""
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for dir_name in dirs:
            dir_path = base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return {dir_name: base_path / dir_name for dir_name in dirs}
    
    @staticmethod
    def get_checkpoint_path(checkpoint_dir, epoch, step=None):
        """Get checkpoint file path"""
        if step is not None:
            filename = f"checkpoint_epoch_{epoch}_step_{step}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
        return Path(checkpoint_dir) / filename
    
    @staticmethod
    def get_latest_checkpoint(checkpoint_dir):
        """Get path to latest checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda x: x.stat().st_mtime)


class DeviceManager:
    """Manage device (CPU/GPU) operations"""
    
    @staticmethod
    def get_device(cuda=True, device_id=0):
        """Get device"""
        if cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
            print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device('cpu')
            print("Using CPU device")
        
        return device
    
    @staticmethod
    def to_device(obj, device):
        """Move object to device"""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: DeviceManager.to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(DeviceManager.to_device(item, device) for item in obj)
        else:
            return obj


class CheckpointManager:
    """Manage model checkpoints"""
    
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, loss, save_path):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    @staticmethod
    def load_checkpoint(model, optimizer, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from epoch {epoch}")
        return model, optimizer, epoch, loss
    
    @staticmethod
    def load_pretrained(model, pretrained_path):
        """Load pretrained weights"""
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
        print(f"Pretrained weights loaded from {pretrained_path}")
        return model


if __name__ == "__main__":
    # Test metrics
    img1 = np.random.rand(256, 256, 3)
    img2 = img1 + np.random.randn(256, 256, 3) * 0.01
    
    psnr = MetricsCalculator.calculate_psnr(img1, img2)
    ssim = MetricsCalculator.calculate_ssim(img1, img2)
    f1, prec, rec = MetricsCalculator.calculate_f1_score(100, 10, 5)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    # Test directory setup
    dirs = PathManager.setup_directories("./test_output")
    print(f"Directories created: {list(dirs.keys())}")
