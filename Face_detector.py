import torch
import torch.nn as nn
from torchvision import transforms

import cv2
import matplotlib.pyplot as plt

from Model_Files.face_detect_model import FaceDetectionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = (224, 224)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE), # temp dataset has a size of 112*92
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

