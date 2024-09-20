# Face detection and identification using OpenCV and CNN(TinyVGG) #

The project is divided into 3 parts, each having a unique role in the finished program

**This project uses 3 main files to perform various functions:**

1. The raw face images obtained first has to be preprocessed, so the Preprocessor.py takes a folder full of images of faces and detects the faces, converts them to grayscale, resizes and normalizes them. then its saved to a directory as mentioned.
2. The face_detect_model.py is the code encompassing the architecture of the model, using the TinyVGG architecture to detect faces. Not a conventionally used architecture to detect faces, but its capable enough to identify faces as long as faces arent too skewed from reality
3. The face_detector.py is the final part that brings together the models state_dict(saved states, parameters, weights etc) and then uses that to predict values for an image that's taken live.

## Model architecture ##

The TinyVGG used incorporate the first block containing a convolution layer followed by another convolution layer, followed by a ReLU activation layer, then followed by a maxpool.
This is replicated in the second block as well the difference being in the input shape. The kernal size and stride for the conv layers are set to default so there is no image resizing until the max pool layer, where the size is cut in half. 

(32, 1, 224, 224) ----> (conv2d(first) -> conv2d(second) -> ReLU -> maxpool)(First conv Block) ----> (32, 10, 112, 112) -> (conv2d(first) -> conv2d(second) -> ReLU -> maxpool)(Second conv Block) ----> (32, 10, 56, 56) -> (flatten() -> nn.Linear)(Linear Block) ----> Number of classes

## Usage ##

### This project has used ###
* Pytorch 2.4
* OpenCV 
* Matplotlib
* Scikit-Learn
* torchvision

### Download and use ###

Download the project files and ensure you have an environment that satisfies the above conditions.

Make sure the path to your folders are all correct
