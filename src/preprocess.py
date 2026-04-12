import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Creates PyTorch DataLoaders for the medical image dataset.
    
    Args:
        data_dir (str): Path to the base dataset directory containing train/val/test folders.
        batch_size (int): Number of images per batch.
        img_size (tuple): Target resize dimension for the CNN.
        
    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Define transformations for grayscale conversion, resizing, and tensor normalized structure.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor() # Converts (H,W,C) into (C,H,W) array ranging from 0.0 to 1.0!
    ])
    
    loaders = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            dataset = datasets.ImageFolder(root=split_path, transform=transform)
            # Shuffle only the training set
            shuffle = True if split == 'train' else False
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            print(f"Warning: Directory {split_path} not found!")

    return loaders

if __name__ == "__main__":
    print("Testing PyTorch dataloaders...")
    loaders = get_data_loaders('../data/chest_xray')
    if 'train' in loaders:
        for images, labels in loaders['train']:
            print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
            print("Successfully loaded exactly 1 batch of images. Setup is perfect!")
            break
