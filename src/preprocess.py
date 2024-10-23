import os
from torchvision import transforms
from PIL import Image

def get_preprocessing_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_data(image):
    transform = get_preprocessing_transform()

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    elif isinstance(image, Image.Image):
        image = image.convert('RGB')

    else:
        raise ValueError('Input must be a file path or a PIL Image')

    return transform(image)

if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(__file__), "..", "raw_data", "training_set")
    image_path = os.path.join(root_dir, "ship", "1__20150830_000652_1_0b07__-122.32370681389538_37.72016772243502.png")
    processed_image = preprocess_data(image_path)

    print(f'Image size before preprocessing: {Image.open(image_path).size}')
    print(f'Image size after preprocessing: {processed_image.shape}')
