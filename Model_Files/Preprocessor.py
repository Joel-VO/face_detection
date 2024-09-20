import cv2
from torchvision import transforms
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')

import os


FOLDER_PATH = "/home/joel/coding/Datasets/drive-download-20240920T135540Z-001/Anna P Joseph "
SAVE_FOLDER = "/home/joel/coding/Datasets/archive/Anna P Joseph"


if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

face_cascade = cv2.CascadeClassifier('/home/joel/coding/open_cv/haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(FOLDER_PATH, filename)

        img = cv2.imread(image_path)


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"No face detected in {filename}")
            continue

        faces = sorted(faces, key=lambda b: b[2] * b[3])
        (x, y, w, h) = faces[-1]
        face = gray[y:y + h, x:x + w]

        face_pil = Image.fromarray(face)
        transformed_face = transform(face_pil)

        save_path = os.path.join(SAVE_FOLDER, f"processed_{filename}")


        transformed_face_pil = transforms.ToPILImage()(transformed_face)
        transformed_face_pil.save(save_path)

        print(f"Processed and saved {filename} to {save_path}")

print("Processing complete!")
