import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running on {device} with {torch.cuda.get_device_name()}")

IMAGE_SIZE = (224, 224)# setting the image size to a value of 224(change as per requirement)

dataset_path = "/home/joel/coding/Datasets/newest augmented dataset"

# transform to be applied to the raw dataset,we resize to the said size,
# and we convert to the image to a tensor and then normalise it

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#ImageFolder is used to load data and label them according to folders, and the transforms applied
dataset = datasets.ImageFolder(root=dataset_path,transform = transform)
classes = dataset.classes

#Doing a train test split according to the image index
size = len(dataset)
indexes = list(range(size))
train_indices, test_indices = train_test_split(indexes, test_size=0.2, random_state=42, stratify=[y for _, y in dataset])
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

#using dataloader to load small batches of data at a time, set batch size to 32
train_batch = DataLoader(train_dataset, batch_size=64,shuffle=True)
test_batch = DataLoader(test_dataset, batch_size=64,shuffle=False)

"""uncomment the below code snippet to get a sample of the image that's in the dataset"""
# for images, labels in train_batch:
#     print(images[0].shape)
#     plt.imshow(images[0].squeeze())
#     plt.title(labels)
#     plt.show()
#     break

class FaceDetectionCNN (nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),#first conv layer, padding of one to ensure same size output
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)# maxpool applied to reduce compuatations and to generalise better
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*56*56,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape) # use this to debug if you're training for more than OUTPUT_SHAPE classes
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# setting up input layers, hidden layers and output layers

INPUT_SHAPE = 3
HIDDEN_SHAPE = 64
label_count = 0
label = 0
for _, labels in dataset:
    if label == labels:
        continue
    else:
        label = labels
        label_count+=1
OUTPUT_SHAPE = label_count+1
# print(OUTPUT_SHAPE) # for debugging

# defining model, optimizer and loss_fn
model = FaceDetectionCNN(INPUT_SHAPE, HIDDEN_SHAPE, OUTPUT_SHAPE).to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def testing( epochs, epoch, learning_model=model, loss_fn = loss_fn, data = test_batch):
    model.eval()
    with torch.inference_mode():
        model.train()
        correct = 0
        running_loss = 0
        total = 0
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = learning_model(images)
            loss = loss_fn(y_pred, labels)

            running_loss += loss.item()
            _, predicted = y_pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(data)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Accuracy: {accuracy:.2f}%, Epoch loss: {epoch_loss:.2f}")

def training( epochs, epoch, learning_model=model, optimizer_fn=optimizer, loss_fn = loss_fn, data = train_batch):
    learning_model.to(device)
    model.train()
    correct = 0
    running_loss = 0
    total = 0
    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        optimizer_fn.zero_grad()

        y_pred = learning_model(images)
        loss = loss_fn(y_pred, labels)

        loss.backward()

        optimizer_fn.step()

        running_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss/len(data)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Accuracy: {accuracy:.2f}%, Epoch loss: {epoch_loss:.2f}")

def main():
    epochs = 10
    for epoch in range(epochs):
        print("Training_accuracy: ")
        training(epochs=epochs, epoch = epoch)
        print("Testing_accuracy: ")
        testing(epochs=epochs, epoch = epoch)

if __name__ == '__main__':
    main()

    PATH = "/home/joel/coding/ide/pycharm/projects/face_detection/Model_Files/Model/model_dict"
    torch.save(model.state_dict(),f=PATH)
    # this model is then used in the face recognition code
