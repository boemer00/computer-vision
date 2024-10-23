import os
import torch
from dataset import preprocess_data

def predict(model, image):
    model.eval()
    with torch.no_grad():
        # preprocess image
        image = preprocess_data(image)

        output = model(image.unsqueeze(0))
        _, predicted_class = torch.max(output, 1)

        class_names = {'no-ship': 0, 'ship': 1}
        predicted_label = class_names[predicted_class.item()]

    return predicted_label
