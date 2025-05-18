from train import train
from test import evaluate
from model import ConvNeXtKAN
from dataset import train_loader, test_loader, class_names
import torch.nn as nn
import torch.optim as optim
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, and Optimizer
model = ConvNeXtKAN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train the model
train(model, train_loader, criterion, optimizer, device)

# Load trained model and test
model.load_state_dict(torch.load("convnext_kan.pth"))
evaluate(model, test_loader, class_names, device)

