import cv2

import torch
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from Model_Files.face_detect_model import FaceDetectionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = (224, 224)
IMAGE_PATH = "/home/joel/coding/Datasets/archive/s26/1.pgm"
img = Image.open(IMAGE_PATH)

# face_cascade = cv2.CascadeClassifier('/home/joel/coding/open_cv/haarcascade_frontalface_default.xml')
# cam = cv2.VideoCapture(0)

# while True:
#     ret, frame = cam.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
#     if len(faces)>1:
#         faces = sorted(faces, key=lambda b: b[2]*b[3])
#         (x, y, w, h) = faces[-1]
#         frame_img = frame[y:y+h, x:x+w]
#         cv2.imwrite(IMAGE_PATH, frame_img)
#         break

# cam.release()
# cv2.destroyAllWindows()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE), # temp dataset has a size of 112*92
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transformed_image = transform(img)

plt.imshow(transformed_image.squeeze(),cmap='gray')
plt.show()

PATH = "/home/joel/coding/ide/pycharm/projects/face_detection/Model_Files/model_dict"


INPUT_SHAPE = IMAGE_SIZE[0]*IMAGE_SIZE[1]
HIDDEN_SHAPE = 10
OUTPUT_SHAPE = 40

model = FaceDetectionCNN(INPUT_SHAPE, HIDDEN_SHAPE, OUTPUT_SHAPE).to(device)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()
transformed_image = transformed_image.to(device)
with torch.inference_mode():
    y_logits = model(transformed_image)
    y_prob = torch.softmax(y_logits,dim=1)
    y_pred = y_logits.argmax(dim=1)
    print(y_pred)
    print(y_prob)