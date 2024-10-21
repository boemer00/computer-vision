import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


root_dir = os.path.join(os.path.dirname(__file__), "..", "raw_data", "training_set")

def get_data_loaders(root_dir, batch_size=16, train_val_split=0.9):
    # resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load data
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    # train and val split
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders(root_dir)
    print(f'Train size: {len(train_loader.dataset)}')
    print(f'Val size: {len(val_loader.dataset)}')
