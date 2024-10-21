import os
import torch
from dataset import get_data_loaders
from initialize_model import initialize_model
from train_model import train_model
from plot import plot_history


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(__file__), "..", "raw_data", "training_set")

    # Load data
    train_loader, val_loader = get_data_loaders(root_dir, batch_size=16)

    # Initialize model
    model = initialize_model(num_classes=2)

    # Train model
    history =train_model(model, train_loader, val_loader, epochs=5)

    # Plot training history
    plot_history(history)
