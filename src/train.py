import os
import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import mlflow.pytorch

from dataset import get_data_loaders
from model import initialize_model
from plot import plot_history
from params import MLFLOW_URI, EXPERIMENT_NAME, PATH_TO_LOCAL_MODEL, EPOCHS, LEARNING_RATE, BATCH_SIZE


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training history
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predict = torch.max(outputs, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        # store training metrics
        history['loss'].append(epoch_loss)
        history['accuracy'].append(accuracy)

        # validate after each epoch
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # log metrics to MLFlow
        mlflow.log_metric('loss', epoch_loss, step=epoch)
        mlflow.log_metric('accuracy', accuracy, step=epoch)
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        mlflow.log_metric('val_accuracy', val_accuracy, step=epoch)

    print('----Finished Training----')
    return history


def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(__file__), "..", "raw_data", "training_set")

    # set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # load data
    train_loader, val_loader = get_data_loaders(root_dir, batch_size=BATCH_SIZE)

    # initialize model
    model = initialize_model(num_classes=2)

    # start MLFlow run
    with mlflow.start_run():
        # log parameters
        mlflow.log_param('epochs', EPOCHS)
        mlflow.log_param('learning_rate', LEARNING_RATE)
        mlflow.log_param('batch_size', BATCH_SIZE)

        # train model
        history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)

        # log the model
        mlflow.pytorch.log_model(model, "model")

        # save the model locally
        torch.save(model.state_dict(), PATH_TO_LOCAL_MODEL)

    # plot training history
    plot_history(history)
