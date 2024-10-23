import torch.nn as nn
from torchvision import models

def initialize_model(num_classes=2):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

if __name__ == "__main__":
    model = initialize_model()
    print(model)
