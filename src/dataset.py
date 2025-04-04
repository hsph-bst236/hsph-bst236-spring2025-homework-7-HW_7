import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

class CustomCIFAR(Dataset):
    """
    A custom CIFAR-10 dataset that inherits from torch.utils.data.Dataset.
    It allows for user-defined augmentations and provides visualization capabilities.
    """
    def __init__(self, train=True, subset_size=None, target_transform=None, mixup_prob=0.4):
        # Store the train parameter as an instance variable
        self.train = train
        
        # Define default transforms if none provided
        if train: # training set
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip() # TODO: Add more data augmentation for training set
            ])
        else: # test set
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
        self.target_transform = target_transform
        self.mixup_prob = mixup_prob  # Store the mixup probability

        # Load the CIFAR-10 dataset with no transform initially
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=None,  # We'll apply transforms manually in __getitem__
            target_transform=self.target_transform
        )

        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                            'dog', 'frog', 'horse', 'ship', 'truck']

        # Optionally use a subset of the dataset.
        if subset_size is not None:
            self.dataset = Subset(self.dataset, range(min(subset_size, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get images without any transforms applied
        if isinstance(self.dataset, Subset):
            img1, label1 = self.dataset.dataset[self.dataset.indices[idx]]
        else:
            img1, label1 = self.dataset[idx]

        # Apply mixup with probability 0.4
        if self.train and random.random() < self.mixup_prob:
            # Pick a second image at random
            # TODO: Your code here  
            
            return mixed_img, mixed_label
        elif self.train:
            # No mixup, just return single transformed image with one-hot label

            #TODO: Your code here
            
            return tensor_img, label_onehot
        else: # test set
            # No mixup, just return single transformed image with one-hot label
            tensor_img = self.transform(img1) # just standardize the image
            
            # Convert label to one-hot
            label_onehot = torch.zeros(len(self.class_names))
            label_onehot[label1] = 1.0
            
            return tensor_img, label_onehot
        
    def get_loader(self, batch_size=64, shuffle=True):
        """
        Returns a DataLoader for the dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def get_train_val_loaders(self, batch_size=64, validation_split=0.2, shuffle=True):
        """
        Returns separate train and validation DataLoaders based on the current dataset.
        
        Args:
            batch_size: Batch size for both loaders
            validation_split: Fraction of the dataset to use for validation
            shuffle: Whether to shuffle the data
            
        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        from torch.utils.data import random_split
        
        # Calculate split sizes
        dataset_size = len(self)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(
            self,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,  # No need to shuffle validation data
        )
        
        return train_loader, val_loader

    def visualize_samples(self, num_samples=8, loader=None):
        """
        Visualizes samples from the dataset or from a provided data loader.
        Assumes normalization of mean=0.5 and std=0.5.
        
        Args:
            num_samples: Number of samples to visualize
            loader: Optional DataLoader to get samples from. If None, samples directly from dataset.
        """
        if loader is not None:
            # Get a batch from the loader
            images, labels = next(iter(loader))
            # Limit to num_samples
            images = images[:num_samples]
            labels = labels[:num_samples]
        else:
            # Collect first num_samples items directly from the dataset
            samples = [self[i] for i in range(num_samples)]
            images, labels = zip(*samples)
            # Convert tuple of tensors to tensor
            images = torch.stack(images)
            labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
        
        # Unnormalize images for display
        images_display = images.clone()  # Create a copy to avoid modifying the original
        images_display = images_display * 0.5 + 0.5  # Unnormalize: [0, 1] range
        
        # Create plot
        fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(min(num_samples, len(images)) * 2, 2))
        if min(num_samples, len(images)) == 1:
            axes = [axes]  # Make sure axes is always a list
        
        for idx, ax in enumerate(axes):
            if idx >= len(images):
                break
            
            # Check if this is a mixup label (one-hot encoded with values between 0 and 1)
            if isinstance(labels[idx], torch.Tensor) and labels[idx].dim() > 0 and labels[idx].numel() > 1:
                # Get top 2 classes for mixup images
                values, indices = torch.topk(labels[idx], 2)
                
                # Only consider it a mixup if second value is significant (> 0.1)
                if values[1] > 0.1:
                    # This is a mixup image
                    class1 = self.class_names[indices[0].item()]
                    class2 = self.class_names[indices[1].item()]
                    ratio = f"{values[0].item():.2f}:{values[1].item():.2f}"
                    title = f"Mix: {class1}/{class2}"
                else:
                    # Mostly one class
                    label_idx = indices[0].item()
                    title = f"{self.class_names[label_idx]}"
            else:
                # Not a one-hot encoded label
                label_idx = labels[idx].item() if isinstance(labels[idx], torch.Tensor) else labels[idx]
                title = f"{self.class_names[label_idx]}"
            
            # Rearrange dimensions for display [C,H,W] -> [H,W,C]
            img_np = images_display[idx].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)  # Ensure values are in valid range
            
            ax.imshow(img_np)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig



if __name__ == "__main__":
    # Create a small subset for quick testing.
    custom_dataset = CustomCIFAR(train=True, subset_size=16)
    loader = custom_dataset.get_loader(batch_size=20)
    for images, labels in loader:
        # Check the shape of the batch
        print("Batch shape:", images.shape)
        print("Batch label shape:", labels.shape)
        break
    # Visualize the samples
    custom_dataset.visualize_samples(num_samples=10, loader=loader)




# %%
