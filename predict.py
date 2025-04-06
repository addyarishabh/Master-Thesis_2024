<<<<<<< HEAD
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ConvNeXtKAN
from dataset import class_names  # Load class names

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtKAN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("convnext_kan.pth"))
model.eval()

# Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    """Loads an image, processes it, and predicts its class with probabilities."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]  # Convert logits to probabilities

    # Get sorted predictions
    sorted_indices = torch.argsort(probabilities, descending=True)
    
    print("\nPredicted Probabilities for All Classes:")
    for idx in sorted_indices:
        class_name = class_names[idx.item()]
        confidence = probabilities[idx].item() * 100
        print(f"{class_name}: {confidence:.2f}%")

    # Get the top predicted class
    top_class = class_names[sorted_indices[0].item()]
    top_confidence = probabilities[sorted_indices[0]].item() * 100

    print(f"\nFinal Prediction: {top_class} ({top_confidence:.2f}%)")
    return top_class, top_confidence

# Example usage:
image_path = r"D:\final_KAN\proliferate eye.jpg"  # Change this path to your test image
predict_image(image_path)
=======
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ConvNeXtKAN
from dataset import class_names  # Load class names

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtKAN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("convnext_kan.pth"))
model.eval()

# Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    """Loads an image, processes it, and predicts its class with probabilities."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]  # Convert logits to probabilities

    # Get sorted predictions
    sorted_indices = torch.argsort(probabilities, descending=True)
    
    print("\nPredicted Probabilities for All Classes:")
    for idx in sorted_indices:
        class_name = class_names[idx.item()]
        confidence = probabilities[idx].item() * 100
        print(f"{class_name}: {confidence:.2f}%")

    # Get the top predicted class
    top_class = class_names[sorted_indices[0].item()]
    top_confidence = probabilities[sorted_indices[0]].item() * 100

    print(f"\nFinal Prediction: {top_class} ({top_confidence:.2f}%)")
    return top_class, top_confidence

# Example usage:
image_path = r"D:\final_KAN\proliferate eye.jpg"  # Change this path to your test image
predict_image(image_path)
>>>>>>> b486782a6c4cdcbbe100f1ccdfade07bd2e310ae
