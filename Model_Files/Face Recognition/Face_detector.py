import cv2

import torch
from torchvision import transforms, datasets
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from Model_Files.Model.face_detect_model import FaceDetectionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = (224, 224)


face_cascade = cv2.CascadeClassifier('/path/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cont = True
while cont:
    print("smile please :) ")
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)#  minSize=(30,30)
        if len(faces)>1:
            faces = sorted(faces, key=lambda b: b[2]*b[3])
            (x, y, w, h) = faces[-1]
            frame_img = frame[y:y+h+8, x:x+w+8]
            break

    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE), # temp dataset has a size of 112*92
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset_path = "/path_to_dataset/"
    dataset = datasets.ImageFolder(root=dataset_path,transform = transform)
    classes = dataset.classes

    INPUT_SHAPE = 3
    HIDDEN_SHAPE = 64
    OUTPUT_SHAPE = 6 #change for different dataset
    # print(OUTPUT_SHAPE)

    face_pil = Image.fromarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))

    transformed_image = transform(face_pil)

    image_for_plot = transformed_image.permute(1, 2, 0)
    image_for_plot = (image_for_plot * 0.5) + 0.5 # it denormalizes the image
    plt.imshow(image_for_plot)
    plt.imshow(image_for_plot.squeeze()) # cmap='gray'
    plt.show()
    transformed_image = transformed_image.unsqueeze(0)

    PATH_dict = "/path_to_the_model/model_dict"
    model = FaceDetectionCNN(INPUT_SHAPE, HIDDEN_SHAPE, OUTPUT_SHAPE)
    model.load_state_dict(torch.load(PATH_dict, weights_only=True))
    model.eval()
    with torch.inference_mode():
        y_logits = model(transformed_image)
        y_prob = torch.softmax(y_logits,dim=1)
        y_pred = y_logits.argmax(dim=1)
        prob_person = y_prob.max(dim=1)
        print(f"The identified person is: {classes[y_pred]} with a probability of {float(prob_person.values)}")
    condn = input("Want to do it again? [y/n]: ")
    if condn in ["Y", "y"]:
        cont = True
    elif condn in ["N", "n"]:
        cont = False

cv2.destroyAllWindows()
cam.release()
