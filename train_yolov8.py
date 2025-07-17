#!/usr/bin/env python3
"""
YOLOv8 Fine-tuning Script for Pizza Store Violation Detection
Trains YOLOv8 on the pizza store dataset to detect: Hand, Person, Pizza, Scooper
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

def load_dataset_config():
    """Load and validate dataset configuration"""
    data_yaml_path = "shared/dataset/data.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset config not found at {data_yaml_path}")
        print("Please ensure you have downloaded the dataset from Roboflow in YOLOv8 format")
        sys.exit(1)
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Dataset Configuration:")
    print(f"   - Classes: {config.get('names', [])}")
    print(f"   - Train: {config.get('train', 'Not found')}")
    print(f"   - Val: {config.get('val', 'Not found')}")
    print(f"   - Test: {config.get('test', 'Not found')}")
    
    return config

def check_dataset_structure():
    """Verify dataset structure exists"""
    required_dirs = [
        "shared/dataset/train/images",
        "shared/dataset/train/labels", 
        "shared/dataset/valid/images",
        "shared/dataset/valid/labels"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("Error: Missing dataset directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\nPlease ensure your dataset is properly extracted from Roboflow")
        sys.exit(1)
    
    print("Dataset structure verified")

def train_yolov8():
    """Main training function"""
    print("Starting YOLOv8 Fine-tuning for Pizza Store Violation Detection")
    print("=" * 70)
    
    # Check dataset
    check_dataset_structure()
    config = load_dataset_config()
    
    # Create model directory if it doesn't exist
    model_dir = Path("shared/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base YOLOv8 model
    print("\nLoading base YOLOv8 model...")
    try:
        # You can choose different sizes: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
        # 'yolov8m.pt' is a good balance between speed and accuracy
        model = YOLO('yolov8m.pt')
        print("Base model loaded successfully")
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("This will automatically download the model from Ultralytics on first run")
        sys.exit(1)
    
    # Training configuration
    training_config = {
        'data': 'shared/dataset/data.yaml',  # Path to dataset config
        'epochs': 10,                        # Number of training epochs
        'imgsz': 640,                        # Input image size
        'batch': 16,                         # Batch size (adjust based on GPU memory)
        'patience': 5,                      # Early stopping patience
        'save': True,                        # Save checkpoints
        'save_period': 10,                   # Save every N epochs
        'cache': False,                      # Cache images for faster training
        'device': 'cpu',                    # Force CPU training
        'workers': 8,                        # Number of worker threads
        'project': 'shared/model',           # Save directory
        'name': 'yolov8m-finetuned',         # Experiment name
        'exist_ok': True,                    # Overwrite existing experiment
        'pretrained': True,                  # Use pretrained weights
        'optimizer': 'auto',                 # Optimizer (SGD, Adam, etc.)
        'verbose': True,                     # Verbose output
        'seed': 42,                          # Random seed for reproducibility
        'deterministic': True,               # Deterministic training
        'single_cls': False,                 # Single class training
        'rect': False,                       # Rectangular training
        'cos_lr': False,                     # Cosine learning rate scheduler
        'close_mosaic': 10,                  # Close mosaic augmentation
        'resume': False,                     # Resume from last checkpoint
        'amp': True,                         # Automatic Mixed Precision
        'fraction': 1.0,                     # Fraction of dataset to use
        'profile': False,                    # Profile ONNX and TensorRT
        'freeze': None,                      # Freeze layers
        'lr0': 0.01,                         # Initial learning rate
        'lrf': 0.01,                         # Final learning rate
        'momentum': 0.937,                   # SGD momentum/Adam beta1
        'weight_decay': 0.0005,              # Optimizer weight decay
        'warmup_epochs': 3.0,                # Warmup epochs
        'warmup_momentum': 0.8,              # Warmup initial momentum
        'warmup_bias_lr': 0.1,               # Warmup initial bias lr
        'box': 7.5,                          # Box loss gain
        'cls': 0.5,                          # Class loss gain
        'dfl': 1.5,                          # DFL loss gain
        'pose': 12.0,                        # Pose loss gain
        'kobj': 2.0,                         # Keypoint obj loss gain
        'label_smoothing': 0.0,              # Label smoothing epsilon
        'nbs': 64,                           # Nominal batch size
        'overlap_mask': True,                # Masks should overlap during training
        'mask_ratio': 4,                     # Mask downsample ratio
        'dropout': 0.0,                      # Use dropout regularization
        'val': True,                         # Validate during training
    }
    
    print(f"\nTraining Configuration:")
    print(f"   - Model: YOLOv8 Medium")
    print(f"   - Epochs: {training_config['epochs']}")
    print(f"   - Image Size: {training_config['imgsz']}")
    print(f"   - Batch Size: {training_config['batch']}")
    print(f"   - Learning Rate: {training_config['lr0']}")
    print(f"   - Save Directory: {training_config['project']}/{training_config['name']}")
    
    # Start training
    print(f"\nStarting training...")
    print("This may take a while depending on your hardware and dataset size.")
    print("You can monitor progress in the terminal and in the runs directory.")
    
    try:
        # Start training
        results = model.train(**training_config)
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {training_config['project']}/{training_config['name']}/weights/")
        print(f"Best model: {training_config['project']}/{training_config['name']}/weights/best.pt")
        print(f"Last model: {training_config['project']}/{training_config['name']}/weights/last.pt")
        
        # Print training results summary
        if results:
            print(f"\nTraining Results Summary:")
            print(f"   - Final mAP50: {results.results_dict.get('metrics/mAP50', 'N/A')}")
            print(f"   - Final mAP50-95: {results.results_dict.get('metrics/mAP50-95', 'N/A')}")
            print(f"   - Training time: {results.results_dict.get('train/epoch', 'N/A')} epochs")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        print(f"Partial results saved to: {training_config['project']}/{training_config['name']}/")
        return False
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print(f"Common issues:")
        print(f"   - Insufficient GPU memory: Try reducing batch size")
        print(f"   - Dataset issues: Check your data.yaml and image paths")
        print(f"   - Missing dependencies: Run 'pip install -r requirements.txt'")
        return False

def validate_model():
    """Validate the trained model on test set"""
    print(f"\nValidating trained model...")
    
    model_path = "shared/model/yolov8m-finetuned/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        results = model.val(data='shared/dataset/data.yaml')
        print(f"Validation completed")
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Pizza Store Violation Detection - YOLOv8 Training")
    print("=" * 70)
    
    # Check if running from correct directory
    if not os.path.exists("shared/dataset"):
        print("Error: Please run this script from the project root directory")
        print("   Expected: pizza_violation_system/")
        print(f"   Current: {os.getcwd()}")
        sys.exit(1)
    
    # Start training
    success = train_yolov8()
    
    if success:
        # Optional: Validate the model
        validate_choice = input(f"\nWould you like to validate the trained model? (y/n): ").lower()
        if validate_choice in ['y', 'yes']:
            validate_model()
    
    print(f"\nTraining script completed!")
    print(f"Next steps:")
    print(f"   1. Check the trained model in shared/model/yolov8m-finetuned/weights/")
    print(f"   2. Use best.pt in your detection service")
    print(f"   3. Start building your microservices") 