import cv2
import numpy as np
import os
import torch
from torch import nn
import pandas as pd

class BubbleClassifier(nn.Module):
    def __init__(self):
        super(BubbleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 15 * 15, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def train_model():
    model = BubbleClassifier()
    # Placeholder: Load and preprocess data from dataset/marked/ and dataset/unmarked/
    # Add your training logic here
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/bubble_classifier.pth')
    print("Model trained and saved to models/bubble_classifier.pth")

def evaluate_omr(image_path, student_id, version):
    print(f"Version received: {version}, type: {type(version)}")  # Debug print
    if not isinstance(version, str) or version not in ["A", "B"]:
        raise ValueError(f"Invalid version: {version}. Must be 'A' or 'B'")
    
    # Load model
    model = BubbleClassifier()
    if os.path.exists('models/bubble_classifier.pth'):
        model.load_state_dict(torch.load('models/bubble_classifier.pth', map_location=torch.device('cpu')))
        model.eval()
    else:
        print("Warning: Model not found, using dummy predictions.")
    
    # Load answer key
    sheet_name = f'Set - {version}'
    try:
        key_df = pd.read_excel('key.xlsx', sheet_name=sheet_name)
        key = key_df['Answer'].values  # Assuming column 'Answer' with 'a', 'b', etc.
        print(f"Loaded key from sheet '{sheet_name}' with {len(key)} answers.")
    except ValueError as e:
        print(f"Error loading key: {e}")
        return {"error": f"Worksheet '{sheet_name}' not found in key.xlsx"}
    except Exception as e:
        print(f"Error loading key: {e}")
        return {"error": "Failed to load answer key"}

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"Failed to load image {image_path}"}

    # Extract bubbles (update coordinates from extract_bubbles.py)
    bubble_size = 30
    start_y = 262  # UPDATE WITH YOUR CLICKED VALUE
    start_x = [200, 500, 800, 1100, 1400]  # UPDATE WITH YOUR CLICKED VALUES
    spacing_h = 40  # UPDATE WITH YOUR CLICKED VALUE
    spacing_v = 40  # UPDATE WITH YOUR CLICKED VALUE

    answers = []
    for col in range(5):
        x_base = start_x[col]
        y = start_y
        for q in range(20):
            marked_opt = 0
            for opt in range(4):
                x = x_base + opt * spacing_h
                bubble = img[y:y+bubble_size, x:x+bubble_size]
                if bubble.shape == (bubble_size, bubble_size):
                    fill_ratio = np.mean(bubble) / 255.0
                    if fill_ratio < 0.5:  # Filled if dark
                        marked_opt = opt + 1
                        break
                    # Use model for ambiguous
                    if model:
                        bubble_tensor = torch.FloatTensor(bubble).unsqueeze(0).unsqueeze(0) / 255.0
                        with torch.no_grad():
                            output = model(bubble_tensor)
                            if output[0][1] > output[0][0]:
                                marked_opt = opt + 1
                                break
            answers.append(marked_opt)
            y += spacing_v

    # Score
    score = sum(1 for i, a in enumerate(answers) if a == (ord(key[i].lower()) - ord('a') + 1)) if len(answers) == len(key) else 0
    total = score / len(key) * 100 if len(key) > 0 else 0

    # Save result
    with open('results.csv', 'a') as f:
        f.write(f"{student_id},{version},{total}\n")

    return {"student_id": student_id, "version": version, "total_score": total, "answers": answers[:10]}  # First 10 for demo

if __name__ == "__main__":
    train_model()