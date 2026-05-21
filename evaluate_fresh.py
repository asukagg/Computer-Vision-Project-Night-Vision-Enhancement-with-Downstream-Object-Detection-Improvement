"""
Fresh Evaluation of Improved Model
Generate current metrics and visualizations
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import yaml

from data_loader import LOLDataset
from improved_model import ImprovedEnhancer
from utils import CheckpointManager, MetricsCalculator


def load_model(model_class, checkpoint_path, device):
    """Load model from checkpoint"""
    try:
        model = model_class(in_channels=3, out_channels=3).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def enhance_image(model, image, device):
    """Enhance a single image"""
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().unsqueeze(0).to(device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0).to(device)
        
        enhanced = model(image)
        return enhanced.squeeze(0).cpu()


def compute_metrics(enhanced, target):
    """Compute metrics"""
    # Ensure tensors are on CPU and detached
    if enhanced.is_cuda:
        enhanced = enhanced.cpu()
    if target.is_cuda:
        target = target.cpu()
    
    enhanced_np = enhanced.detach().permute(1, 2, 0).numpy().clip(0, 1)
    target_np = target.detach().permute(1, 2, 0).numpy().clip(0, 1)
    
    # PSNR
    psnr = peak_signal_noise_ratio(target_np, enhanced_np, data_range=1.0)
    
    # SSIM
    ssim = structural_similarity(target_np, enhanced_np, data_range=1.0, channel_axis=2)
    
    # L1 Loss
    l1 = np.mean(np.abs(enhanced_np - target_np))
    
    # Brightness
    brightness = np.mean(enhanced_np)
    
    # Contrast
    contrast = np.std(enhanced_np)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'l1': l1,
        'brightness': brightness,
        'contrast': contrast
    }


def main():
    print("=" * 80)
    print("FRESH EVALUATION - IMPROVED NIGHT VISION ENHANCEMENT")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        eval_dataset = LOLDataset(config['dataset']['lol_root'], split='val', image_size=256)
        print(f"✓ Validation dataset: {len(eval_dataset)} samples")
    except:
        eval_dataset = LOLDataset(config['dataset']['lol_root'], split='train', image_size=256)
        print(f"✓ Training dataset: {len(eval_dataset)} samples")
    
    # Load model
    print("\nLoading improved model...")
    model = load_model(ImprovedEnhancer, 'checkpoints/checkpoint_best.pth', device)
    if model is None:
        print("✗ Failed to load model")
        return
    print("✓ Model loaded")
    
    # Evaluate
    print(f"\nEvaluating on 100 samples...")
    metrics = {'psnr': [], 'ssim': [], 'l1': [], 'brightness': [], 'contrast': []}
    
    sample_indices = np.random.choice(len(eval_dataset), min(100, len(eval_dataset)), replace=False)
    
    for idx in tqdm(sample_indices):
        try:
            batch_data = eval_dataset[idx]
            if isinstance(batch_data, dict):
                low_light = batch_data.get('low', batch_data.get('input')).to(device)
                high_light = batch_data.get('high', batch_data.get('target')).to(device)
            else:
                low_light, high_light = batch_data
                low_light = low_light.to(device)
                high_light = high_light.to(device)
            
            enhanced = enhance_image(model, low_light, device)
            m = compute_metrics(enhanced, high_light.cpu())
            for key in metrics:
                metrics[key].append(m[key])
        except:
            continue
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    metrics_info = [
        ('PSNR (dB)', 'psnr'),
        ('SSIM', 'ssim'),
        ('L1 Loss', 'l1'),
        ('Brightness', 'brightness'),
        ('Contrast', 'contrast')
    ]
    
    for name, key in metrics_info:
        mean_val = np.mean(metrics[key])
        std_val = np.std(metrics[key])
        print(f"{name:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs('comparisons', exist_ok=True)
    
    # Metrics dashboard - Clean visualization
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Main metrics display - larger format
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.axis('off')
    
    psnr_val = np.mean(metrics['psnr'])
    ssim_val = np.mean(metrics['ssim'])
    l1_val = np.mean(metrics['l1'])
    brightness_val = np.mean(metrics['brightness'])
    contrast_val = np.mean(metrics['contrast'])
    
    main_text = f"""
IMPROVED MODEL - NIGHT VISION ENHANCEMENT METRICS

    PSNR:         {psnr_val:.2f} dB  (Higher is better)
    SSIM:         {ssim_val:.4f}    (Closer to 1.0 is better)
    L1 Loss:      {l1_val:.4f}      (Lower is better)
    Brightness:   {brightness_val:.4f}      (Perceptual quality)
    Contrast:     {contrast_val:.4f}      (Details preservation)
    """
    
    ax_main.text(0.05, 0.5, main_text, fontsize=13, verticalalignment='center',
                fontfamily='monospace', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#E8F4F8', edgecolor='#2C3E50', linewidth=2, alpha=0.9))
    
    # Individual metric panels with distribution
    metric_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, (name, key) in enumerate(metrics_info[:4]):
        row, col = metric_positions[idx]
        ax = fig.add_subplot(gs[row, col])
        mean_val = np.mean(metrics[key])
        std_val = np.std(metrics[key])
        values = metrics[key]
        
        # Histogram
        ax.hist(values, bins=15, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.4f}')
        
        ax.set_xlabel(name, fontweight='bold', fontsize=11)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=10)
        ax.set_title(f'{name} Distribution\nμ={mean_val:.4f} σ={std_val:.4f}', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.savefig('comparisons/metrics_evaluation.png', dpi=150, bbox_inches='tight')
    print("✓ Metrics visualization saved: comparisons/metrics_evaluation.png")
    plt.close()
    
    # Generate comparison images
    print("Generating comparison images...")
    sample_indices_viz = np.random.choice(len(eval_dataset), min(10, len(eval_dataset)), replace=False)
    
    for img_idx, sample_idx in enumerate(sample_indices_viz):
        try:
            batch_data = eval_dataset[sample_idx]
            if isinstance(batch_data, dict):
                low_light = batch_data.get('low', batch_data.get('input')).to(device)
                high_light = batch_data.get('high', batch_data.get('target')).to(device)
            else:
                low_light, high_light = batch_data
                low_light = low_light.to(device)
                high_light = high_light.to(device)
            
            enhanced = enhance_image(model, low_light, device)
            
            low_np = (low_light.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            high_np = (high_light.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            enh_np = (enhanced.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            
            # 2x2 grid
            h, w = low_np.shape[:2]
            comparison = np.zeros((h*2, w*2, 3), dtype=np.uint8)
            comparison[0:h, 0:w] = low_np
            comparison[0:h, w:2*w] = enh_np
            comparison[h:2*h, 0:w] = high_np
            comparison[h:2*h, w:2*w] = enh_np
            
            cv2.putText(comparison, 'Input', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(comparison, 'Enhanced', (w+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(comparison, 'Reference', (10, h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out_path = f'comparisons/comparison_{img_idx:02d}.png'
            cv2.imwrite(out_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        except:
            continue
    
    print(f"✓ Generated 10 comparison images in comparisons/")
    
    print("\n" + "=" * 80)
    print("✓ FRESH EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
