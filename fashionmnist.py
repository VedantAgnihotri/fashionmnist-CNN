import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(root=r"C:\Users\vedant agnihotri\Desktop\Code\Python\ML-DL\pytorch\neuralnetwork\computer vision",
                                    train=True, 
                                    download=True, 
                                    transform=transform, 
                                    target_transform=None)

test_data = datasets.FashionMNIST(root=r"C:\Users\vedant agnihotri\Desktop\Code\Python\ML-DL\pytorch\neuralnetwork\computer vision",
                                  train=False,
                                  transform=transform,
                                  download=True,
                                  target_transform=None)

device = 'cuda'
class_names = train_data.classes

train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)


class modelv0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels = 32, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=128*7*7, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=len(class_names))
        )
    def forward(self, x)->torch.tensor:
        return self.layer_stack(x)
    
model = modelv0().to(device)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)

train_loss = []
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        if batch % 100 == 99:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch+1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}')
            train_loss.append(running_loss/100)
            running_loss = 0.0


model.eval()
with torch.inference_mode():
    total = 0
    correct = 0
    for (X, y) in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        _, predicted = torch.max(y_pred, 1)
        total += y.size(0)  
        correct += (predicted == y).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
print(f"total: {total}, correct: {correct}")

plt.plot(train_loss, label="Training Loss")
plt.xlabel('Iterations (every 100 batches)')
plt.ylabel('Loss')
plt.legend()
plt.show()