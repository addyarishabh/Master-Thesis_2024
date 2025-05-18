<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import ConvNeXtKAN
from dataset import train_loader, class_names

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, and Optimizer
model = ConvNeXtKAN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training function
def train(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%')

    torch.save(model.state_dict(), "convnext_kan.pth")
    print("Model saved as 'convnext_kan.pth'")

# Run training
train(model, train_loader, criterion, optimizer, device)

