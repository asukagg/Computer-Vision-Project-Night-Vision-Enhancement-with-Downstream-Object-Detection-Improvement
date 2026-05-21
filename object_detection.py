"""
Object Detection for Enhanced LOL Images
Detect and label objects in enhanced night vision images
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

try:
    import yolov5
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("⚠ YOLOv5 not installed. Install with: pip install yolov5")

from exdark_classifier import ObjectClassifier, ExDarkDataset
from improved_model import ImprovedEnhancer


class ObjectDetectionPipeline:
    """Detect and label objects in enhanced images"""
    
    def __init__(self, exdark_classifier_path=None, yolo_model='yolov5s'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ExDark classifier
        if exdark_classifier_path and os.path.exists(exdark_classifier_path):
            self.exdark_model = ObjectClassifier(num_classes=12).to(self.device)
            self.exdark_model.load_state_dict(torch.load(exdark_classifier_path, map_location=self.device))
            self.exdark_model.eval()
            
            # Class mapping
            self.exdark_classes = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
                                   'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
            print("✓ ExDark classifier loaded")
        else:
            self.exdark_model = None
            print("⚠ ExDark classifier not found - using YOLO only")
        
        # Load YOLO model if available
        if YOLO_AVAILABLE:
            try:
                import logging
                logging.getLogger('yolov5').setLevel(logging.WARNING)
                # Use torch.hub to load YOLOv5
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.yolo_model.to(self.device)
                self.yolo_model.conf = 0.45  # Set confidence threshold
                print(f"✓ YOLOv5 (yolov5s) loaded")
            except Exception as e:
                self.yolo_model = None
                print(f"⚠ Failed to load YOLO model: {str(e)[:100]}")
        else:
            self.yolo_model = None
    
    def detect_objects_yolo(self, image):
        """Detect objects using YOLO"""
        if self.yolo_model is None:
            return []
        
        results = self.yolo_model(image)
        detections = []
        
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = float(conf)
            class_idx = int(cls)
            label = results.names[class_idx]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'label': label,
                'source': 'YOLO'
            })
        
        return detections
    
    def detect_objects_exdark(self, image, patch_size=224, stride=112, confidence_threshold=0.5):
        """Detect ExDark objects using sliding window"""
        if self.exdark_model is None:
            return []
        
        h, w = image.shape[:2]
        detections = []
        
        # Sliding window detection
        with torch.no_grad():
            for y in range(0, h - patch_size, stride):
                for x in range(0, w - patch_size, stride):
                    patch = image[y:y+patch_size, x:x+patch_size]
                    
                    # Prepare patch
                    patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # Predict
                    output = self.exdark_model(patch_tensor)
                    confidence, class_idx = torch.max(torch.softmax(output, 1), 1)
                    
                    if confidence.item() > confidence_threshold:
                        detections.append({
                            'bbox': (x, y, x + patch_size, y + patch_size),
                            'confidence': confidence.item(),
                            'label': self.exdark_classes[class_idx.item()],
                            'source': 'ExDark'
                        })
        
        return detections
    
    def detect(self, image):
        """Detect objects using both YOLO and ExDark"""
        detections = []
        
        # YOLO detection
        if self.yolo_model:
            detections.extend(self.detect_objects_yolo(image))
        
        # ExDark detection
        if self.exdark_model:
            detections.extend(self.detect_objects_exdark(image))
        
        return detections
    
    def draw_detections(self, image, detections, show_confidence=True):
        """Draw bounding boxes and labels on image"""
        output = image.copy()
        
        colors = {
            'YOLO': (0, 255, 0),      # Green
            'ExDark': (255, 165, 0)   # Orange
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            confidence = det['confidence']
            source = det['source']
            color = colors.get(source, (255, 0, 0))
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_confidence:
                text = f"{label} ({confidence:.2f}) [{source}]"
            else:
                text = f"{label} [{source}]"
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(output, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(output, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output


def process_enhanced_images(enhanced_dir, output_dir, classifier_path=None):
    """Process enhanced LOL images and detect objects"""
    
    print("=" * 80)
    print("OBJECT DETECTION ON ENHANCED IMAGES")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = ObjectDetectionPipeline(exdark_classifier_path=classifier_path)
    
    # Process images
    image_files = list(Path(enhanced_dir).glob('*.jpg')) + list(Path(enhanced_dir).glob('*.png'))
    print(f"\n✓ Found {len(image_files)} images to process")
    
    detection_stats = {
        'total_images': len(image_files),
        'images_with_detections': 0,
        'total_detections': 0,
        'yolo_detections': 0,
        'exdark_detections': 0,
        'object_distribution': {}
    }
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Detect objects
            detections = detector.detect(image)
            
            if detections:
                detection_stats['images_with_detections'] += 1
                detection_stats['total_detections'] += len(detections)
                
                for det in detections:
                    if det['source'] == 'YOLO':
                        detection_stats['yolo_detections'] += 1
                    else:
                        detection_stats['exdark_detections'] += 1
                    
                    label = det['label']
                    detection_stats['object_distribution'][label] = detection_stats['object_distribution'].get(label, 0) + 1
                
                # Draw and save
                labeled_image = detector.draw_detections(image, detections, show_confidence=True)
                
                output_path = os.path.join(output_dir, f"detected_{img_path.stem}.jpg")
                cv2.imwrite(output_path, labeled_image)
        
        except Exception as e:
            print(f"  ⚠ Error processing {img_path.name}: {str(e)}")
    
    return detection_stats


if __name__ == "__main__":
    # This would be called after training classifier
    # For now, placeholder
    print("Object detection module loaded")
