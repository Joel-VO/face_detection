import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_path = "/home/joel/coding/Datasets/archive"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)), # temp dataset has a size of 112*92
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.ImageFolder(root=dataset_path,transform = transform)

size = len(dataset)
indexes = list(range(size))
train_indices, test_indices = train_test_split(indexes, test_size=0.2, random_state=42, stratify=[y for _, y in dataset])
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_batch = DataLoader(train_dataset, batch_size=32,shuffle=True)
test_batch = DataLoader(test_dataset, batch_size=32,shuffle=False)

for images, labels in train_batch:
    plt.imshow(images[0].squeeze())
    plt.title(labels)
    plt.show()
    break

class FaceDetectionCNN (nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer(x)


model = FaceDetectionCNN(50176, 10, 40).to(device)
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
    epochs = 100
    for epoch in range(epochs):
        training(epochs=epochs, epoch = epoch)
        testing(epochs=epochs, epoch = epoch)

if __name__ == '__main__':
    main()

