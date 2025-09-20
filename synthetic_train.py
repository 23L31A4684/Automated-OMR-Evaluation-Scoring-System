import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

class BubbleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images / 255.0, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def generate_bubble(is_marked, noise_level=0.1):
    img = np.ones((32, 32)) * 255
    center = (16, 16)
    radius = 10
    yy, xx = np.mgrid[:32, :32]
    distances = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    outline = (distances >= radius - 1) & (distances <= radius + 1)
    img[outline] = 0
    if is_marked:
        fill_intensity = np.random.uniform(0.3, 1.0)
        base_fill = (1 - fill_intensity) * 50
        mask = distances <= radius
        noise = np.random.normal(0, noise_level * 50, img.shape)
        img[mask] = base_fill + noise[mask]
    img += np.random.normal(0, noise_level * 20, img.shape)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# Generate dataset
data = []
labels = []
np.random.seed(42)
for _ in range(1000):
    data.append(generate_bubble(False, np.random.uniform(0.05, 0.2)))
    labels.append(0)
for _ in range(1000):
    data.append(generate_bubble(True, np.random.uniform(0.05, 0.2)))
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

indices = np.arange(len(data))
np.random.shuffle(indices)
split = int(0.8 * len(data))
train_idx = indices[:split]
test_idx = indices[split:]
X_train = data[train_idx][:, np.newaxis, :, :]
y_train = labels[train_idx]
X_test = data[test_idx][:, np.newaxis, :, :]
y_test = labels[test_idx]

train_ds = BubbleDataset(X_train, y_train)
test_ds = BubbleDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

class BubbleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 2)  # Updated to 4096 (16x16x16)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten to batch_size x 4096
        x = self.fc1(x)
        return x

model = BubbleClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Training Accuracy: {accuracy}%")

if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/bubble_classifier.pth')
print("Model trained and saved to models/bubble_classifier.pth")