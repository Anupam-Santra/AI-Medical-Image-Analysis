import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        
        # Convolutional Block 1
        # Input shape: (Batch Size, 1, 224, 224). 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        # Input shape shrinks to 112x112 after pool1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        # Generates 56x56
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4
        # Generates 28x28
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After pool4, the tensor size is 128 channels * 14 * 14 = 25088 features
        self.flatten = nn.Flatten()
        
        # Fully Connected Classifier
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        # Final output layer: 1 neuron for binary classification (Normal=0, Pneumonia=1)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Pass data through conv blocks with ReLU activation
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # Flatten for Dense Layers
        x = self.flatten(x)
        
        # Pass through dense classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Note: We don't apply sigmoid here because we will use BCEWithLogitsLoss during training
        return x

def build_model():
    """Return a fresh instance of the CNN model."""
    return PneumoniaCNN()

if __name__ == "__main__":
    # Test model shape consistency
    model = build_model()
    # Create a dummy batch of 2 grayscale images (2, 1, 224, 224)
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    print(f"Model successfully loaded. Output batch shape: {output.shape}")
