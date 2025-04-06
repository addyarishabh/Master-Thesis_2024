<<<<<<< HEAD
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define dataset directory
DATASET_PATH = r'D:\final_KAN\image'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load entire dataset
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Define train-test split (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Class names (from dataset)
class_names = full_dataset.classes  # ['Healthy', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferate DR']
=======
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define dataset directory
DATASET_PATH = r'D:\final_KAN\image'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load entire dataset
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Define train-test split (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Class names (from dataset)
class_names = full_dataset.classes  # ['Healthy', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferate DR']
>>>>>>> b486782a6c4cdcbbe100f1ccdfade07bd2e310ae
