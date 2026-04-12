import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import get_data_loaders
from model import build_model
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

def evaluate_model():
    """Evaluates the saved model on the test set and generates a confusion matrix."""
    print("Loading Test Data...")
    loaders = get_data_loaders(DATA_DIR, batch_size=32)
    
    if 'test' not in loaders:
        print(f"Test data not found. Please ensure Kaggle dataset is correctly extracted to {DATA_DIR}")
        return

    model_path = os.path.join(MODELS_DIR, 'pneumonia_cnn_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run 'train.py' first.")
        return

    print("Checking device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading Saved Model...")
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Generating Predictions...")
    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            preds = (probs > 0.5).astype(int).reshape(-1)
            all_preds.extend(preds)
            all_labels.extend(labels)

    y_test = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'], 
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    matrix_path = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
    plt.savefig(matrix_path)
    print(f"Confusion matrix saved to '{matrix_path}'")

def predict_single_image(img_path):
    """Utility to predict a single image like a doctor would in simulation."""
    if not os.path.exists(img_path):
        # Fallback check if user provided relative path from root but executing from src
        fallback_path = os.path.join(BASE_DIR, img_path)
        if os.path.exists(fallback_path):
            img_path = fallback_path
        else:
            print(f"Image not found at {img_path}")
            return

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = build_model()
        model_path = os.path.join(MODELS_DIR, 'pneumonia_cnn_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        img_arr = cv2.imread(img_path)
        tensor_img = transform(img_arr).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            prediction = torch.sigmoid(output).item()
            
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

        print(f"--- AI DIAGNOSIS ---")
        print(f"Result: {label}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict_single_image(sys.argv[1])
    else:
        evaluate_model()
