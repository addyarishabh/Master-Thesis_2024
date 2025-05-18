<<<<<<< HEAD
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from model import ConvNeXtKAN
from dataset import test_loader, class_names

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtKAN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("convnext_kan.pth"))
model.eval()

# Compute Sensitivity & Specificity
def compute_metrics(conf_matrix):
    """Compute class-wise and overall sensitivity and specificity from confusion matrix."""
    num_classes = conf_matrix.shape[0]
    
    # True Positives, False Positives, False Negatives, True Negatives
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    # Compute Sensitivity (Recall) and Specificity for each class
    sensitivity_per_class = TP / (TP + FN)
    specificity_per_class = TN / (TN + FP)

    # Compute Overall Sensitivity and Specificity (Macro Average)
    overall_sensitivity = np.mean(sensitivity_per_class)
    overall_specificity = np.mean(specificity_per_class)

    return sensitivity_per_class, specificity_per_class, overall_sensitivity, overall_specificity

# Compute AUC-ROC Curve
def plot_roc_curve(y_true, y_probs, class_names, filename='roc_curve.png'):
    """Plots and saves the AUC-ROC curve for each class."""
    plt.figure(figsize=(8, 6))
    
    auc_scores = []
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        auc = roc_auc_score(y_true == i, y_probs[:, i])
        auc_scores.append(auc)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()

    return auc_scores

# Model Evaluation
def evaluate(model, test_loader, class_names, device):
    """Evaluate model and compute metrics."""
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Compute loss and accuracy
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100

    # Compute Sensitivity, Specificity & AUC Scores
    conf_matrix = confusion_matrix(all_labels, all_preds)
    sensitivity_per_class, specificity_per_class, overall_sensitivity, overall_specificity = compute_metrics(conf_matrix)
    auc_scores = plot_roc_curve(np.array(all_labels), np.array(all_probs), class_names)

    print(f'Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f'Overall Sensitivity: {overall_sensitivity:.4f}')
    print(f'Overall Specificity: {overall_specificity:.4f}')

    # Display Class-wise Sensitivity, Specificity & AUC
    print("\nClass-wise Sensitivity, Specificity & AUC Scores:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Sensitivity = {sensitivity_per_class[i]:.4f}, "
              f"Specificity = {specificity_per_class[i]:.4f}, "
              f"AUC = {auc_scores[i]:.4f}")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix, class_names)

# Confusion Matrix Plot
def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()

# Run Evaluation
evaluate(model, test_loader, class_names, device)
