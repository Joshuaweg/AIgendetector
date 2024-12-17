from classifier import *
from interpret import load_video
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the model
def load_model_for_inference(checkpoint_path, device):
    # Initialize model
    latentEncoder = LatentEncoder()
    patchEncoder = PatchEncoder()
    classifier = Classifier()
    model = VideoClassifier(latentEncoder, patchEncoder, classifier).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load only model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model

classes = ["AI-Generated", "Real"]
latentEncoder = LatentEncoder()
patchEncoder = PatchEncoder()
classifier = Classifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vclf = load_model_for_inference('model\\videoClassifier_epoch_11.pth', device)



# Path to list of videos
path_to_list_of_videos = 'data\\test_paths.txt'

# Initialize lists to store predictions and true labels
all_preds = []
all_labels = []

# Load the list of videos
with open(path_to_list_of_videos, 'r') as f:
    lines = f.readlines()
    # Iterate through the list of videos and run the model on each
    for line in lines:
        video_path = line.strip()
        frames_tensor, label, video_path = load_video(video_path)
        frames_tensor = frames_tensor.unsqueeze(0).to("cuda")
       
        # Forward pass through the model
        with torch.no_grad():
            outputs = vclf(frames_tensor)
       
        # Get the predicted class
        pred = torch.argmax(outputs, dim=1).item()
       
        # Append predictions and true labels to the lists
        all_preds.append(pred)
        all_labels.append(label)

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Create a visual confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_11.png')
plt.close()

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

# Convert classification report to DataFrame
df_report = pd.DataFrame(report).transpose()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Hide axes
ax.axis('off')

# Create table
table = ax.table(cellText=df_report.values.round(3),
                 rowLabels=df_report.index,
                 colLabels=df_report.columns,
                 cellLoc='center',
                 loc='center')

# Modify table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color coding
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
for i, key in enumerate(df_report.index):
    for j in range(len(df_report.columns)):
        cell = table[i+1, j]
        if j < 3:  # Apply only to precision, recall, f1-score columns
            cell_value = df_report.iloc[i, j]
            cell_color = colors[int(cell_value * 10) - 1] if not pd.isna(cell_value) else 'white'
            cell.set_facecolor(cell_color)

plt.title('Classification Report', fontsize=16)
plt.tight_layout()
plt.savefig('improved_classification_report+epoch_11.png', dpi=300, bbox_inches='tight')
plt.close()

print("Improved visual classification report has been saved as 'improved_classification_report.png'")

print("Confusion Matrix:")
print(conf_matrix)
print(f"{classes[0]}|{classes[1]}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\nVisual outputs have been saved as 'confusion_matrix.png' and 'classification_report.html'")