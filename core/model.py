import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A robust, modular Convolutional Neural Network suitable for standard image classification.
    Supports both MNIST (1-channel, 28x28) and CIFAR-10 (3-channel, 32x32) datasets.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10, input_size: int = 32):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Block 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Conv -> ReLU -> Pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        # Compute fully connected input dimension dynamically
        # input size is halved 3 times (e.g. 32 -> 16 -> 8 -> 4)
        fc_features = 128 * (input_size // 8) * (input_size // 8)
        
        self.fc1 = nn.Linear(fc_features, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(dataset_name: str = "cifar10") -> SimpleCNN:
    """Helper function to load the correct CNN configuration based on dataset name."""
    if dataset_name.lower() == "mnist":
        return SimpleCNN(in_channels=1, num_classes=10, input_size=28)
    elif dataset_name.lower() == "cifar10":
        return SimpleCNN(in_channels=3, num_classes=10, input_size=32)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'mnist' or 'cifar10'.")
