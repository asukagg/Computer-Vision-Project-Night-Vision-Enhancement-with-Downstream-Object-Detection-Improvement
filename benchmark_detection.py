"""
Comprehensive Benchmarking and Comparison
Compare enhancement quality with object detection results
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json

from improved_model import ImprovedEnhancer
from data_loader import LOLDataset
from object_detection import ObjectDetectionPipeline


class ComprehensiveBenchmark:
    """Generate comprehensive benchmarks and comparisons"""
    
    def __init__(self, enhancement_checkpoint, classifier_checkpoint=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load enhancement model
        self.enhancement_model = ImprovedEnhancer(3, 3).to(self.device)
        checkpoint = torch.load(enhancement_checkpoint, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.enhancement_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.enhancement_model.load_state_dict(checkpoint)
        self.enhancement_model.eval()
        print("✓ Enhancement model loaded")
        
        # Load detection model
        self.detector = ObjectDetectionPipeline(exdark_classifier_path=classifier_checkpoint)
    
    def enhance_image_tensor(self, low_light_tensor):
        """Enhance image from tensor"""
        with torch.no_grad():
            enhanced = self.enhancement_model(low_light_tensor.unsqueeze(0).to(self.device))
            return enhanced.squeeze(0).cpu()
    
    def compute_enhancement_metrics(self, low_light_np, enhanced_np, reference_np):
        """Compute image quality metrics"""
        # Normalize to [0, 1]
        low_light_np = np.clip(low_light_np, 0, 1)
        enhanced_np = np.clip(enhanced_np, 0, 1)
        reference_np = np.clip(reference_np, 0, 1)
        
        metrics = {
            'psnr': peak_signal_noise_ratio(reference_np, enhanced_np, data_range=1.0),
            'ssim': structural_similarity(reference_np, enhanced_np, data_range=1.0, channel_axis=2),
            'l1_loss': np.mean(np.abs(enhanced_np - reference_np)),
            'brightness_enhanced': np.mean(enhanced_np),
            'brightness_low': np.mean(low_light_np),
            'brightness_ref': np.mean(reference_np),
            'contrast_enhanced': np.std(enhanced_np),
            'contrast_low': np.std(low_light_np),
            'contrast_ref': np.std(reference_np),
        }
        
        return metrics
    
    def benchmark_with_detection(self, num_samples=50):
        """Generate comprehensive benchmark with detection results"""
        
        print("=" * 80)
        print("COMPREHENSIVE BENCHMARK - ENHANCEMENT + DETECTION")
        print("=" * 80)
        
        # Load LOL dataset
        print("\nLoading LOL validation dataset...")
        try:
            dataset = LOLDataset('LOLdataset', split='val', image_size=256)
        except:
            dataset = LOLDataset('LOLdataset', split='train', image_size=256)
        
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        print(f"✓ Processing {len(sample_indices)} samples")
        
        # Benchmark results
        enhancement_metrics = {
            'psnr': [],
            'ssim': [],
            'l1_loss': [],
            'brightness_improvement': [],
            'contrast_improvement': []
        }
        
        detection_metrics = {
            'detections_low': [],
            'detections_enhanced': [],
            'object_types': {},
            'detection_confidence': []
        }
        
        print("\nBenchmarking...")
        
        for idx in tqdm(sample_indices):
            try:
                batch_data = dataset[idx]
                if isinstance(batch_data, dict):
                    low_light = batch_data.get('low', batch_data.get('input'))
                    reference = batch_data.get('high', batch_data.get('target'))
                else:
                    low_light, reference = batch_data
                
                # Enhance
                enhanced = self.enhance_image_tensor(low_light)
                
                # Compute metrics
                low_np = low_light.permute(1, 2, 0).numpy()
                enh_np = enhanced.permute(1, 2, 0).numpy()
                ref_np = reference.permute(1, 2, 0).numpy()
                
                metrics = self.compute_enhancement_metrics(low_np, enh_np, ref_np)
                
                for key in ['psnr', 'ssim', 'l1_loss']:
                    enhancement_metrics[key].append(metrics[key])
                
                brightness_improve = (metrics['brightness_enhanced'] - metrics['brightness_low']) / (metrics['brightness_ref'] - metrics['brightness_low'] + 1e-6)
                enhancement_metrics['brightness_improvement'].append(brightness_improve)
                
                contrast_improve = metrics['contrast_enhanced'] / (metrics['contrast_low'] + 1e-6)
                enhancement_metrics['contrast_improvement'].append(contrast_improve)
                
                # Detection on low-light and enhanced
                low_cv = (low_np * 255).astype(np.uint8)
                enh_cv = (enh_np * 255).astype(np.uint8)
                low_cv = cv2.cvtColor(low_cv, cv2.COLOR_RGB2BGR)
                enh_cv = cv2.cvtColor(enh_cv, cv2.COLOR_RGB2BGR)
                
                det_low = self.detector.detect(low_cv)
                det_enh = self.detector.detect(enh_cv)
                
                detection_metrics['detections_low'].append(len(det_low))
                detection_metrics['detections_enhanced'].append(len(det_enh))
                
                for det in det_enh:
                    label = det['label']
                    detection_metrics['object_types'][label] = detection_metrics['object_types'].get(label, 0) + 1
                    detection_metrics['detection_confidence'].append(det['confidence'])
            
            except Exception as e:
                continue
        
        # Aggregate results
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        print("\n📊 ENHANCEMENT QUALITY METRICS:")
        print(f"  PSNR:                {np.mean(enhancement_metrics['psnr']):.2f} ± {np.std(enhancement_metrics['psnr']):.2f} dB")
        print(f"  SSIM:                {np.mean(enhancement_metrics['ssim']):.4f} ± {np.std(enhancement_metrics['ssim']):.4f}")
        print(f"  L1 Loss:             {np.mean(enhancement_metrics['l1_loss']):.4f} ± {np.std(enhancement_metrics['l1_loss']):.4f}")
        print(f"  Brightness Improve:  {np.mean(enhancement_metrics['brightness_improvement']):.2f}x")
        print(f"  Contrast Improve:    {np.mean(enhancement_metrics['contrast_improvement']):.2f}x")
        
        print("\n🔍 OBJECT DETECTION METRICS:")
        print(f"  Avg detections (low-light):    {np.mean(detection_metrics['detections_low']):.1f}")
        print(f"  Avg detections (enhanced):     {np.mean(detection_metrics['detections_enhanced']):.1f}")
        print(f"  Detection improvement:         {(np.mean(detection_metrics['detections_enhanced']) / (np.mean(detection_metrics['detections_low']) + 1e-6) - 1) * 100:.1f}%")
        print(f"  Avg detection confidence:      {np.mean(detection_metrics['detection_confidence']):.2f}")
        
        print("\n📦 OBJECT TYPES DETECTED:")
        for obj_type, count in sorted(detection_metrics['object_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {obj_type:15s}: {count:3d} detections")
        
        return enhancement_metrics, detection_metrics
    
    def generate_comparison_report(self, enhancement_metrics, detection_metrics, output_dir='comparisons'):
        """Generate visual report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Row 1: Enhancement metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(['PSNR'], [np.mean(enhancement_metrics['psnr'])], yerr=[np.std(enhancement_metrics['psnr'])], 
                color='#FF6B6B', alpha=0.7, capsize=5)
        ax1.set_ylabel('dB')
        ax1.set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(['SSIM'], [np.mean(enhancement_metrics['ssim'])], yerr=[np.std(enhancement_metrics['ssim'])],
                color='#4ECDC4', alpha=0.7, capsize=5)
        ax2.set_ylabel('Score')
        ax2.set_title('Structural Similarity Index', fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(['L1 Loss'], [np.mean(enhancement_metrics['l1_loss'])], yerr=[np.std(enhancement_metrics['l1_loss'])],
                color='#45B7D1', alpha=0.7, capsize=5)
        ax3.set_ylabel('Loss')
        ax3.set_title('L1 Reconstruction Loss', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Row 2: Improvement metrics
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(['Brightness'], [np.mean(enhancement_metrics['brightness_improvement'])],
                color='#FFA07A', alpha=0.7)
        ax4.set_ylabel('Improvement Factor')
        ax4.set_title('Brightness Improvement', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(['Contrast'], [np.mean(enhancement_metrics['contrast_improvement'])],
                color='#98D8C8', alpha=0.7)
        ax5.set_ylabel('Improvement Factor')
        ax5.set_title('Contrast Improvement', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        det_low = np.mean(detection_metrics['detections_low'])
        det_enh = np.mean(detection_metrics['detections_enhanced'])
        improvement = (det_enh / (det_low + 1e-6) - 1) * 100
        
        ax6.bar(['Detection\nImprovement'], [improvement], color='#95E1D3', alpha=0.7)
        ax6.set_ylabel('Improvement (%)')
        ax6.set_title('Detection Improvement', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        # Row 3: Detection comparison
        ax7 = fig.add_subplot(gs[2, 0])
        bars = ax7.bar(['Low-Light', 'Enhanced'], [det_low, det_enh], color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax7.set_ylabel('Avg Detections per Image')
        ax7.set_title('Detection Count Comparison', fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        # Set proper y-axis limits
        max_det = max(det_low, det_enh) if max(det_low, det_enh) > 0 else 1
        ax7.set_ylim([0, max_det * 1.3])
        
        # Object type distribution
        ax8 = fig.add_subplot(gs[2, 1:3])
        obj_types = sorted(detection_metrics['object_types'].items(), key=lambda x: x[1], reverse=True)
        if obj_types:
            labels, counts = zip(*obj_types[:8])
            ax8.barh(labels, counts, color='#F38181', alpha=0.7)
            ax8.set_xlabel('Detection Count')
            ax8.set_title('Object Types Detected (Top 8)', fontweight='bold')
            ax8.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Comprehensive Benchmark: Enhancement + Object Detection', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(output_dir, 'benchmark_report.png'), dpi=150, bbox_inches='tight')
        print(f"\n✓ Benchmark report saved: {output_dir}/benchmark_report.png")
        plt.close()


def main():
    """Run comprehensive benchmark"""
    
    benchmark = ComprehensiveBenchmark(
        enhancement_checkpoint='checkpoints/checkpoint_best.pth',
        classifier_checkpoint='checkpoints/exdark_classifier.pth'
    )
    
    enhancement_metrics, detection_metrics = benchmark.benchmark_with_detection(num_samples=50)
    benchmark.generate_comparison_report(enhancement_metrics, detection_metrics)


if __name__ == "__main__":
    main()
