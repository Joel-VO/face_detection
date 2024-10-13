import cv2
from torchvision import transforms
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')# used for linux in wayland use cases

import os

FOLDER_PATH = "path to input dir"
SAVE_FOLDER = "path to output dir"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

face_cascade = cv2.CascadeClassifier('/path/haarcascade_frontalface_default.xml')


transform = transforms.Compose([# resizes and converts to tensor
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(FOLDER_PATH, filename)

        img = cv2.imread(image_path)

        # Detect faces on the color image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"No face detected in {filename}")
            continue

        # Process only the largest face
        faces = sorted(faces, key=lambda b: b[2] * b[3])
        (x, y, w, h) = faces[-1]
        face = img[y:y + h, x:x + w]

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))# converts the image to RGB and then to array format
        transformed_face = transform(face_pil)

        save_path = os.path.join(SAVE_FOLDER, f"processed_{filename}")

        transformed_face_pil = transforms.ToPILImage()(transformed_face)
        transformed_face_pil.save(save_path)

        print(f"Processed and saved {filename} to {save_path}")

print("Processing complete!")
