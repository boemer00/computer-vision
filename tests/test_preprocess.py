import pytest
import torch
from PIL import Image
from preprocess import preprocess_data

@pytest.fixture
def sample_image():
    return Image.new('RGB', (80, 80), color='red')

def test_preprocess_data(sample_image):
    # apply the preprocess_data transformation
    transformed_image = preprocess_data(sample_image)

    # check if the output is a tensor
    assert isinstance(transformed_image, torch.Tensor)

    # check the shape of the output tensor
    assert transformed_image.shape == torch.Size([3, 224, 224])
