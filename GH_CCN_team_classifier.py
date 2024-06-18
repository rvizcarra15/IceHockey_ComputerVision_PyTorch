# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:29:51 2024

@author: Raúl Vizcarra Chirinos
"""
# ************CONVOLUTIONAL NEURAL NETWORK-THREE CLASSES DETECTION**************************
# REFEREE
# WHITE TEAM (white_away)
# YELLOW TEAM (yellow_home)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


#******************************Data transformation********************************************
# Training and Validation Datasets
data_dir = 'D:/PYTHON/teams_sample_dataset'

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#********************************CNN Model Architecture**************************************
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)  #Three Classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x
    
    
#********************************CNN TRAINING**********************************************

# Model-loss function-optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#*********************************Training*************************************************
num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.type(torch.LongTensor)  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            labels = labels.type(torch.LongTensor)  
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)  
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            
#********************************METRICS & PERFORMANCE************************************
    
    val_losses.append(val_loss / len(val_loader))
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Loss: {train_losses[-1]:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, "
          f"Val Acc: {val_accuracy:.2%}, "
          f"Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, "
          f"Val F1 Score: {val_f1:.4f}")
    
#*******************************SHOW METRICS & PERFORMANCE**********************************
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()

# SAVE THE MODEL FOR THE GH_CV_track_teams CODE
torch.save(model.state_dict(), 'D:/PYTHON/hockey_team_classifier.pth')


# ********************TEST MODEL PREDICTIONS WITH SAMPLE DATASET***************************

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# SAMPLE DATASET FOR VALIDATION
test_dir = 'D:/PYTHON/validation_dataset'

# CNN MODEL FOR TEAM PREDICTIONS
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)  # THREE CLASS DETECTION
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x

# CNN MODEL PREVIOUSLY SAVED
model = CNNModel()
model.load_state_dict(torch.load('D:/PYTHON/hockey_team_classifier.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#******************ITERATION ON SAMPLE IMAGES-ACCURACY TEST*****************************

class_names = ['team_referee', 'team_away', 'team_home']

def predict_image(image_path, model, transform):
# LOADS DATASET
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  

# MAKES PREDICTIONS
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  
        team = class_names[predicted.item()]
    return team

for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)
    if os.path.isfile(image_path):  
        predicted_team = predict_image(image_path, model, transform)
        print(f'Image {image_name}: The player belongs to {predicted_team}')
        
        




