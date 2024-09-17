import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)), # temp dataset has a size of 112*92
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_path = "/home/joel/coding/Datasets/archive"

train_data = datasets.ImageFolder(root=dataset_path,transform = transform)

train_batch = DataLoader(train_data, batch_size=32,shuffle=True)
count=0
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
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer(x)


model = FaceDetectionCNN(50176, 10, 40)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def training(learning_model=model, optimzer_fn=optimizer, loss_fn = loss_fn, epochs=100, data = train_batch):
    learning_model.to(device)
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            y_pred = learning_model(images)
            loss = loss_fn(y_pred, labels)

            loss.backward()

            optimizer.step()
            _, predicted = y_pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Accuracy: {accuracy:.2f}%")


# training()

