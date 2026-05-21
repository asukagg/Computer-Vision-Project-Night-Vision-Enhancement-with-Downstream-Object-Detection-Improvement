"""
Detect and Label Enhanced Images
Enhance LOL dataset images, run object detection, and save labeled results
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from improved_model import ImprovedEnhancer
from data_loader import LOLDataset
from object_detection import ObjectDetectionPipeline


def enhance_and_detect(num_samples=50, output_dir='detections'):
    """Enhance LOL images and apply object detection with labels"""
    
    print("=" * 80)
    print("ENHANCE AND LABEL WITH OBJECT DETECTION")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Load enhancement model
    print("\nLoading enhancement model...")
    enhancement_model = ImprovedEnhancer(3, 3).to(device)
    checkpoint = torch.load('checkpoints/checkpoint_best.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        enhancement_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        enhancement_model.load_state_dict(checkpoint)
    enhancement_model.eval()
    print("✓ Enhancement model loaded")
    
    # Load detection pipeline
    print("Loading detection pipeline...")
    detector = ObjectDetectionPipeline(exdark_classifier_path='checkpoints/exdark_classifier.pth')
    print("✓ Detection pipeline ready")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/enhanced', exist_ok=True)
    os.makedirs(f'{output_dir}/labeled', exist_ok=True)
    
    # Load LOL dataset
    print("\nLoading LOL validation dataset...")
    try:
        dataset = LOLDataset('LOLdataset', split='val', image_size=256)
    except:
        dataset = LOLDataset('LOLdataset', split='train', image_size=256)
    
    print(f"✓ Dataset loaded: {len(dataset)} images")
    
    # Sample random indices
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    print(f"✓ Processing {len(sample_indices)} samples")
    
    print("\n" + "=" * 80)
    print("PROCESSING IMAGES")
    print("=" * 80)
    
    for idx_in_batch, dataset_idx in enumerate(tqdm(sample_indices, desc="Processing")):
        try:
            # Load image pair
            batch_data = dataset[dataset_idx]
            if isinstance(batch_data, dict):
                low_light = batch_data.get('low', batch_data.get('input'))
                reference = batch_data.get('high', batch_data.get('target'))
            else:
                low_light, reference = batch_data
            
            # Enhance image
            with torch.no_grad():
                enhanced = enhancement_model(low_light.unsqueeze(0).to(device))
                enhanced = enhanced.squeeze(0).cpu()
            
            # Convert to OpenCV format
            enhanced_np = enhanced.permute(1, 2, 0).numpy()
            enhanced_cv = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)
            enhanced_cv = cv2.cvtColor(enhanced_cv, cv2.COLOR_RGB2BGR)
            
            # Run detection
            detections = detector.detect(enhanced_cv)
            
            # Draw detections
            labeled_image = enhanced_cv.copy()
            
            # Draw bounding boxes and labels
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = det['label']
                confidence = det['confidence']
                source = det['source']
                
                # Color based on source
                if source == 'YOLO':
                    color = (0, 255, 0)  # Green
                else:
                    color = (255, 165, 0)  # Orange
                
                # Draw box
                cv2.rectangle(labeled_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                text = f"{label} ({confidence:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # Get text size for background
                (text_width, text_height) = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                # Draw background rectangle
                cv2.rectangle(labeled_image, 
                            (x1, y1 - text_height - 4), 
                            (x1 + text_width + 2, y1), 
                            color, -1)
                
                # Put text
                cv2.putText(labeled_image, text, (x1 + 1, y1 - 3), 
                          font, font_scale, (255, 255, 255), font_thickness)
            
            # Save enhanced image
            enhanced_path = os.path.join(output_dir, 'enhanced', f'enhanced_{idx_in_batch:04d}.jpg')
            cv2.imwrite(enhanced_path, enhanced_cv)
            
            # Save labeled image
            labeled_path = os.path.join(output_dir, 'labeled', f'labeled_{idx_in_batch:04d}.jpg')
            cv2.imwrite(labeled_path, labeled_image)
            
        except Exception as e:
            print(f"\n⚠ Error processing image {dataset_idx}: {str(e)[:100]}")
            continue
    
    print("\n" + "=" * 80)
    print("✓ DETECTION AND LABELING COMPLETE")
    print("=" * 80)
    print(f"\n📁 Output saved to: {output_dir}/")
    print(f"   ├── enhanced/  ({len(sample_indices)} enhanced images)")
    print(f"   └── labeled/   ({len(sample_indices)} labeled images with detections)")
    
    # Generate summary visualization
    print("\nGenerating summary visualization...")
    generate_summary_grid(output_dir, num_samples=min(8, len(sample_indices)))
    print("✓ Summary visualization saved")


def generate_summary_grid(output_dir, num_samples=8):
    """Generate a grid visualization of labeled images"""
    
    labeled_dir = os.path.join(output_dir, 'labeled')
    labeled_images = sorted([f for f in os.listdir(labeled_dir) if f.endswith('.jpg')])[:num_samples]
    
    if not labeled_images:
        return
    
    # Create grid
    cols = min(4, len(labeled_images))
    rows = (len(labeled_images) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, img_name in enumerate(labeled_images):
        row = idx // cols
        col = idx % cols
        
        img_path = os.path.join(labeled_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"Labeled Detection #{idx+1}", fontweight='bold')
        axes[row][col].axis('off')
    
    # Hide empty subplots
    for idx in range(len(labeled_images), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'detection_summary_grid.png')
    plt.savefig(grid_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Grid visualization saved: {grid_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance and detect objects in LOL images')
    parser.add_argument('--num-samples', type=int, default=50, 
                       help='Number of images to process (default: 50)')
    parser.add_argument('--output-dir', type=str, default='detections',
                       help='Output directory for results (default: detections)')
    
    args = parser.parse_args()
    
    enhance_and_detect(num_samples=args.num_samples, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
