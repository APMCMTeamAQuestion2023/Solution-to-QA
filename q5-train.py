import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from model import Classification

device = torch.device('cuda')
image_size = 270
CHANNELS_IMG = 3

print('Initializing model')
model = Classification(in_channels=3, out_channels=5)
model.to(device)
transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
])

print('Trying to load data')
dataset = ImageFolder(root='data/Attachment2_new', transform=transform)
test_dataset = ImageFolder(root='data/Attachment3', transform=transform)
train_ratio = 0.8
dataset_size = len(dataset)
train_size = int(dataset_size * train_ratio)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print('Moving data from Memory to CUDA')
train_dataloader = [(inputs.to(device), labels.to(device)) for inputs, labels in train_dataloader]
val_dataloader = [(inputs.to(device), labels.to(device)) for inputs, labels in val_dataloader]
test_dataloader = [(inputs.to(device), labels.to(device)) for inputs, labels in test_dataloader]
print('Data moved to CUDA.')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

num_epochs = 20

print('Start training')
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item()}")

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(val_dataloader)
    accuracy = 100.0 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {test_loss}, Accuracy: {accuracy}%")

    torch.save(model, f'model_{epoch+1}.pt')
    print('Model saved.')

print('Start testing')
x = list(range(1, 20706))
y = []
with torch.no_grad():
    model.eval()
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        print(outputs)
        _, predicted = outputs.max(1)
        predicted = predicted.tolist()
        y.extend(predicted)

results = pd.DataFrame({'id':x, 'predicted':y})
results.to_csv('q5-results.csv', index=False)