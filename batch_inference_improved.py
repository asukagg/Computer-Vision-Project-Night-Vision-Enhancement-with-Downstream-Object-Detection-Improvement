"""
Batch Inference with Improved Model
Enhanced night vision enhancement for production use
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import argparse
import yaml

from improved_model import ImprovedEnhancer


def load_improved_model(checkpoint_path='checkpoints/checkpoint_best_improved.pth', device=None):
    """Load the improved model from checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading improved model from: {checkpoint_path}")
    
    try:
        model = ImprovedEnhancer(in_channels=3, out_channels=3).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {device}")
        return model, device
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, device


def enhance_image(image, model, device):
    """Enhance a single image with the improved model"""
    with torch.no_grad():
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            # Normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            image_tensor = image.to(device)
        
        # Forward pass
        enhanced = model(image_tensor)
        
        # Convert back to numpy
        enhanced_np = enhanced.squeeze(0).cpu().permute(1, 2, 0).numpy()
        enhanced_np = np.clip(enhanced_np, 0, 1)
        
        # Convert to uint8 for output
        enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
        
        return enhanced_uint8


def enhance_single_image(input_path, output_path, model, device):
    """Enhance a single image and save"""
    try:
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"✗ Failed to read: {input_path}")
            return False
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance
        enhanced = enhance_image(image_rgb, model, device)
        
        # Convert back to BGR for saving
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and output_dir != '.':
            os.makedirs(output_dir, exist_ok=True)
        
        # Save
        cv2.imwrite(output_path, enhanced_bgr)
        return True
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False


def enhance_folder(input_dir, output_dir, model, device, recursive=False):
    """Enhance all images in a folder"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    if recursive:
        image_files = list(Path(input_dir).rglob('*.jpg')) + \
                     list(Path(input_dir).rglob('*.jpeg')) + \
                     list(Path(input_dir).rglob('*.png')) + \
                     list(Path(input_dir).rglob('*.bmp'))
    else:
        image_files = list(Path(input_dir).glob('*.jpg')) + \
                     list(Path(input_dir).glob('*.jpeg')) + \
                     list(Path(input_dir).glob('*.png')) + \
                     list(Path(input_dir).glob('*.bmp'))
    
    if not image_files:
        print(f"✗ No images found in {input_dir}")
        return 0
    
    print(f"\n✓ Found {len(image_files)} images")
    print(f"✓ Processing and saving to: {output_dir}\n")
    
    success_count = 0
    
    for image_path in tqdm(image_files, desc="Enhancing"):
        # Calculate output path
        rel_path = image_path.relative_to(input_dir) if recursive else image_path.name
        output_path = os.path.join(output_dir, str(rel_path))
        
        if enhance_single_image(str(image_path), output_path, model, device):
            success_count += 1
    
    print(f"\n✓ Successfully enhanced {success_count}/{len(image_files)} images")
    print(f"✓ Results saved to: {output_dir}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Batch inference with improved night vision enhancement model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance single image
  python batch_inference_improved.py --input input.jpg --output output.jpg
  
  # Enhance folder
  python batch_inference_improved.py --input ./images_low --output ./images_enhanced
  
  # Enhance folder recursively
  python batch_inference_improved.py --input ./data --output ./enhanced --recursive
  
  # Use custom checkpoint
  python batch_inference_improved.py --input input.jpg --output output.jpg \\
    --checkpoint ./custom_checkpoint.pth
        """)
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image file or directory')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/checkpoint_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--recursive', action='store_true',
                       help='Process folders recursively')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("IMPROVED NIGHT VISION ENHANCEMENT - BATCH INFERENCE")
    print("=" * 80)
    
    # Load model
    device = torch.device(args.device)
    model, device = load_improved_model(args.checkpoint, device)
    
    if model is None:
        print("✗ Failed to load model")
        return
    
    # Process
    if os.path.isfile(args.input):
        # Single image
        print(f"\nEnhancing single image: {args.input}")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        
        if enhance_single_image(args.input, args.output, model, device):
            print(f"✓ Enhanced image saved: {args.output}")
        else:
            print("✗ Failed to enhance image")
    
    elif os.path.isdir(args.input):
        # Folder
        print(f"\nEnhancing folder: {args.input}")
        if args.recursive:
            print("(recursive mode enabled)")
        
        enhance_folder(args.input, args.output, model, device, 
                      recursive=args.recursive)
    
    else:
        print(f"✗ Input path not found: {args.input}")
    
    print("\n" + "=" * 80)
    print("✓ BATCH INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
