import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from preprocess import get_data_loaders
from model import build_model

# Robust absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

def plot_training_history(history, save_path=None):
    """Plots and saves the training accuracy and loss curves."""
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, 'training_history.png')
        
    epochs_range = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(save_path)
    print(f"Training history graph saved to {save_path}")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single pass over the training data."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
        labels = labels.unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total

def validate_model(model, dataloader, criterion, device):
    """Evaluates the model on validation data without updating weights."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            labels = labels.unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / total, correct / total

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Checking for hardware accelerators...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    print("Loading Data...")
    loaders = get_data_loaders(DATA_DIR, batch_size=32)
    
    if 'train' not in loaders or 'val' not in loaders:
        print(f"Data not found. Ensure Kaggle dataset is exactly at: {DATA_DIR}")
        return

    print("Building Model...")
    model = build_model().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    best_val_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("Starting Training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, loaders['val'], criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = os.path.join(MODELS_DIR, 'pneumonia_cnn_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved new best model with Val Acc: {val_acc:.4f}")

    print("Training Complete! Plotting statistics...")
    plot_training_history(history)

if __name__ == "__main__":
    main()
