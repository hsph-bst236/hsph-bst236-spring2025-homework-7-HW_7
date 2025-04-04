import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from model import TinyVGG_Residual
from config import config_TinyVGG
import os
import pandas as pd
import glob
import numpy as np

def test_kaggle_data(model_path, test_dir='.', batch_size=64):
    """
    Test a trained model on the Kaggle CIFAR-10 test dataset.
    
    Args:
        model_path: Path to the saved model checkpoint
        test_dir: Directory containing test PNG images
        batch_size: Batch size for testing
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms - same as used in test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load model
    model = TinyVGG_Residual(
        input_channels=config_TinyVGG["input_channels"],
        num_classes=config_TinyVGG["num_classes"]
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluation mode
    model.eval()
    
    # Class names in the correct order
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get all PNG files from the test directory
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.png')))
    if not test_files:
        raise FileNotFoundError(f"No PNG files found in {test_dir}")
    
    print(f"Found {len(test_files)} test images")
    
    # Process images and make predictions
    predictions = []
    ids = []
    
    # Process in batches for efficiency
    with torch.no_grad():
        for i in range(0, len(test_files), batch_size):
            batch_files = test_files[i:i+batch_size]
            batch_images = []
            batch_ids = []
            
            for file_path in batch_files:
                # Extract ID from filename (assuming filename is ID.png)
                file_id = os.path.basename(file_path).split('.')[0]
                batch_ids.append(file_id)
                
                # Load and preprocess image
                image = Image.open(file_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
                batch_images.append(image_tensor)
            
            # Stack tensors into a batch
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            
            # Forward pass
            outputs = model(batch_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            # Convert tensor predictions to list
            batch_predictions = predicted.cpu().numpy().tolist()
            
            # Add to overall results
            predictions.extend(batch_predictions)
            ids.extend(batch_ids)
            
            # Report progress
            print(f"Processed {min(i+batch_size, len(test_files))}/{len(test_files)} images")
    
    # Convert numeric predictions to class names
    label_names = [class_names[pred] for pred in predictions]
    
    # Create dataframe and save to CSV
    df = pd.DataFrame({
        "id": ids,
        "label": label_names
    })
    
    # Sort by ID (important for Kaggle submission)
    df = df.sort_values(by="id")
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save to CSV
    csv_path = 'results/kaggle_submission.csv'
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

if __name__ == "__main__":
    # Update this path to your best model checkpoint
    model_path = "models/checkpoints/hw7-cifar10-tinyvgg_20250403_090258_best.pth"
    test_dir = 'cifar-10/test'
    
    # Add debug information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking for test files in: {os.path.abspath(test_dir)}")
    
    # First ensure the test directory exists
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
        print(f"Created directory: {test_dir}")
        print("Please extract test.7z to this directory")
        exit(1)
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    else:
        test_kaggle_data(model_path, test_dir=test_dir, batch_size=64)
