from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ConvNeXtKAN  # Import your model class

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = r"D:\final_KAN\convnext_kan.pth"  # Update with the correct path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels
class_labels = ["Healthy", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Load model
model = ConvNeXtKAN(num_classes=5)  # Ensure num_classes matches your training
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess the image
        image = preprocess_image(filepath)

        # Model inference
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()

        prediction = class_labels[predicted_class]  # Get class label
        probability_scores = {class_labels[i]: round(probabilities[i].item(), 4) for i in range(len(class_labels))}

        return render_template("result.html", filename=filename, prediction=prediction, probability_scores=probability_scores)



if __name__ == "__main__":
    app.run(debug=True)



if __name__ == "__main__":
    app.run(debug=True)
