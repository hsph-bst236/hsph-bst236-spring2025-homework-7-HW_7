import torch
import torch.nn as nn
from dataset import CustomCIFAR
from model import TinyVGG_Residual
from utils import compute_confusion_matrix
from config import config_TinyVGG
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def test_model(model_path, batch_size=64, visualize=False, save_predictions=True):
    """
    Test a trained model on the CIFAR-10 test dataset.
    
    Args:
        model_path: Path to the saved model checkpoint
        batch_size: Batch size for testing
        visualize: Whether to visualize confusion matrix
        save_predictions: Whether to save predictions to CSV
    
    Returns:
        accuracy: Test accuracy
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    test_dataset = CustomCIFAR(train=False)
    test_loader = test_dataset.get_loader(batch_size=batch_size, shuffle=False)
    class_names = test_dataset.class_names
    
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
    
    # Initialize metrics
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    predictions = []
    
    # Test loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # For CSV output
            test_label = np.argmax(outputs.cpu().data.numpy(), axis=1)
            predictions.extend(test_label.tolist())
            
            # Handle one-hot encoded labels if necessary
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # Convert one-hot encoded labels to class indices
                target_classes = torch.argmax(labels, dim=1)
            else:
                target_classes = labels
            
            # Accumulate metrics
            total += target_classes.size(0)
            correct += (predicted == target_classes).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target_classes.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Generate confusion matrix
    if visualize:
        # Instead of using the utility function directly with numpy arrays,
        # calculate and visualize the confusion matrix here
        cm = confusion_matrix(all_targets, all_preds)
        
        plt.figure(figsize=(10, 8))
        # Plot confusion matrix
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig('results/confusion_matrix.png')
        plt.close()
    
    # Save predictions to CSV file
    if save_predictions:
        # Create dataframe and save to CSV
        df = pd.DataFrame()
        # Use "id" instead of "Id" for the column name
        df["id"] = [i+1 for i in range(len(predictions))]
        
        # Convert numeric predictions to class names
        label_names = [class_names[pred] for pred in predictions]
        # Use "label" instead of "Category" for the column name
        df["label"] = label_names
        
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Save to CSV
        csv_path = 'results/submission.csv'
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
    
    return accuracy

if __name__ == "__main__":
    # Update this path to your best model checkpoint
    model_path = "models/checkpoints/your_best_model.pth"
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    else:
        test_model(model_path, batch_size=64, visualize=False, save_predictions=True) 