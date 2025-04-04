import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from model import TinyVGG_Residual
from config import config_TinyVGG

class ActivationMaximization:
    """
    Class for performing activation maximization to visualize what features
    a CNN has learned for each class.
    """
    def __init__(self, model_path, device=None, image_size=(32, 32), num_classes=10):
        """
        Initialize with a pre-trained model.
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to run optimization on
            image_size: Size of the generated images
            num_classes: Number of output classes
        """
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Image parameters
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Load the model
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                            'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Define preprocessing for visualization
        self.preprocess = transforms.Compose([
            transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))
        ])
        
    def _load_model(self, model_path):
        """Load the saved model from checkpoint."""
        model = TinyVGG_Residual(
            input_channels=config_TinyVGG["input_channels"],
            num_classes=config_TinyVGG["num_classes"]
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        return model
    
    def total_variation_loss(self, img):
        """
        Calculate total variation loss for regularization.
        This encourages spatial smoothness in the generated image.
        
        Args:
            img: Input tensor image
            
        Returns:
            Total variation loss
        """
        # Calculate differences along height and width dimensions
        h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return h_tv + w_tv
    
    def generate_class_visualization(self, target_class, iterations=300, lr=0.1, tv_weight=0.001):
        """
        Generate an image that maximizes the activation of a specific class.
        
        Args:
            target_class: Class index to visualize
            iterations: Number of optimization iterations
            lr: Learning rate for optimization
            tv_weight: Weight of total variation regularization
            
        Returns:
            Generated image as a numpy array
        """
        # Start with random noise image
        img = torch.randn(1, 3, *self.image_size, requires_grad=True, device=self.device)
        
        # Initialize optimizer
        optimizer = optim.Adam([img], lr=lr)
        
        # Keep track of the progress
        loss_history = []
        img_history = []
        
        # Optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(img)
            
            #TODO: Calculate loss: negative of target class score plus TV regularization
            # Apply softmax to convert logits to probabilities
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            class_score = softmax_outputs[0, target_class]
            tv_loss = self.total_variation_loss(img)
            loss = -class_score + tv_weight * tv_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Store loss
            loss_history.append(loss.item())
            
            # Save image every 50 iterations for visualization
            if i % 50 == 0 or i == iterations - 1:
                # Convert to displayable image
                img_np = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, -1, 1)
                img_np = (img_np + 1) / 2  # Scale from [-1, 1] to [0, 1]
                img_history.append(img_np.copy())
                
                # Print progress
                print(f"Iteration {i}/{iterations}, Loss: {loss.item():.4f}, "
                      f"Class Score: {class_score.item():.4f}, TV Loss: {tv_loss.item():.4f}")
                
        # Return the optimized image and history
        return img_history[-1], img_history, loss_history
    
    def generate_all_classes(self, iterations=300, lr=0.1, tv_weight=0.001, save_dir='results/activation_max'):
        """
        Generate visualizations for all classes.
        
        Args:
            iterations: Number of optimization iterations
            lr: Learning rate for optimization
            tv_weight: Weight of total variation regularization
            save_dir: Directory to save generated images
            
        Returns:
            Dictionary mapping class indices to generated images
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        class_visualizations = {}
        
        # Generate visualization for each class
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            print(f"\nGenerating visualization for class {class_idx}: {class_name}")
            
            # Generate the visualization
            final_img, img_history, loss_history = self.generate_class_visualization(
                class_idx, iterations, lr, tv_weight
            )
            
            # Store the result
            class_visualizations[class_idx] = final_img
            
            # Save the final image
            plt.figure(figsize=(5, 5))
            plt.imshow(final_img)
            plt.title(f"Class {class_idx}: {class_name}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/class_{class_idx}_{class_name}.png", dpi=150)
            
            # Save the optimization progress
            plt.figure(figsize=(15, 5))
            
            # Plot loss history
            plt.subplot(1, 2, 1)
            plt.plot(loss_history)
            plt.title('Loss During Optimization')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            
            # Plot image evolution (first, middle, last)
            plt.subplot(1, 2, 2)
            grid_size = min(len(img_history), 5)
            for i in range(grid_size):
                plt.subplot(1, grid_size, i + 1)
                iteration = i * (iterations // (grid_size - 1)) if grid_size > 1 else 0
                if i == grid_size - 1:
                    iteration = iterations - 1
                plt.imshow(img_history[i])
                plt.title(f"Iter {iteration}")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/progress_{class_idx}_{class_name}.png", dpi=150)
            plt.close('all')
            
        return class_visualizations
    
    def visualize_all_classes(self, class_visualizations=None, save_path=None):
        """
        Create a grid visualization of all class visualizations.
        
        Args:
            class_visualizations: Dictionary of class visualizations.
                                  If None, will be generated.
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure with all class visualizations
        """
        if class_visualizations is None:
            class_visualizations = self.generate_all_classes()
            
        # Create a grid of visualizations
        rows = int(np.ceil(self.num_classes / 5))
        cols = min(5, self.num_classes)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for class_idx in range(self.num_classes):
            ax = axes[class_idx]
            ax.imshow(class_visualizations[class_idx])
            ax.set_title(f"{self.class_names[class_idx]}")
            ax.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # TODO: change the checkpoint path to the best model checkpoint
    checkpoint_path = "models/checkpoints/cifar10-tinyvgg-hwtest_20250402_134356_best.pth"
    
    # Create activation maximization object
    act_max = ActivationMaximization(checkpoint_path)
    
    # Generate and visualize all classes
    # TODO: change the hyperparameters here to generate better results
    visualizations = act_max.generate_all_classes(
        iterations=1000,
        lr=0.1,
        tv_weight=0.001,
        save_dir="results/activation_max"
    )
    
    # Create a combined visualization
    act_max.visualize_all_classes(
        class_visualizations=visualizations,
        save_path="results/activation_max/all_classes.png"
    )
    
    print(f"\nResults saved to results/activation_max/all_classes.png") 