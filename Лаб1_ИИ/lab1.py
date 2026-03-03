import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import random
import torchvision
import torch
import torch.nn as nn

train_dataset = []
train_target = []
test_dataset = []
test_target = []
valid_dataset = []
valid_target = []

path_to_class={}

count = 0
dirs_copy = []
for idx, (root, dirs, files) in enumerate(os.walk("archive/simpsons_dataset")):
    class_name = os.path.basename(root)
    if(idx != 0):
        path_to_class[class_name] = idx-1
    train_part = 0
    test_part = 0
    valid_part = 0
    if(len(files) != 3 or len(files) != 7):
        train_part = int((len(files)*0.75)/1.0)
        test_part = int(((len(files)-train_part)*0.5)/1.0)
        valid_part = int((len(files) - train_part - test_part))
    elif(len(files) == 7):
        train_part = 5
        test_part = 1
        valid_part = 1
    elif(len(files) == 3):
        train_part = 1
        test_part = 1
        valid_part = 1

    for file in files[:train_part]:
        train_dataset.append(os.path.join(root, file))
        train_target.append(idx-1)
    for file in files[train_part:(train_part+test_part)]:
        test_dataset.append(os.path.join(root, file))
        test_target.append(idx-1)
    for file in files[(train_part+test_part):(train_part+test_part+valid_part)]:
        valid_dataset.append(os.path.join(root, file))
        valid_target.append(idx-1)

# Добавить валидационную выборку
class SimpsonDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transform(image)
        
        target = self.targets[idx]
        
        return image, target

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(), 
    torchvision.transforms.Resize((244, 244)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset_obj = SimpsonDataset(train_dataset, train_target, transform=preprocess)
test_dataset_obj = SimpsonDataset(test_dataset, test_target, transform=preprocess)
valid_dataset_obj = SimpsonDataset(valid_dataset, valid_target, transform=preprocess)

train_loader = torch.utils.data.DataLoader(
    train_dataset_obj, 
    batch_size=64, 
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset_obj, 
    batch_size=64, 
    shuffle=False,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset_obj, 
    batch_size=64, 
    shuffle=False,
)

class Simpson_classifi(nn.Module):
    def __init__(self):
        super().__init__()
        self.details = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(), 
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.MaxPool2d(kernel_size=2),
    
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
    
            nn.AdaptiveMaxPool2d((1, 1)),    #Разенков
        )
        self.classifi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(path_to_class)),
        )
    def forward(self, x):
        x = self.details(x)
        x = self.classifi(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Simpson_classifi().to(device)
criterion = nn.CrossEntropyLoss()
optimizator = torch.optim.Adam(model.parameters(), lr=0.005)  #Моментум и RMSProm 

best_valid_accuracy = 0.0
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizator.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizator.step()
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
    model.eval()
    correct_valid = 0
    total_valid = 0
    valid_loss = 0.0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()
    
    valid_accuracy = 100 * correct_valid / total_valid
    avg_valid_loss = valid_loss / len(valid_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')
    
    if valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizator.state_dict(),
            'valid_accuracy': valid_accuracy,
            'valid_loss': avg_valid_loss,
        }
        torch.save(checkpoint, 'checkpoints/best_model.pth')
        print(f'Лучшая модель сохранена с точностью на валидации: {valid_accuracy:.2f}%')

checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f'\nЗагружена лучшая модель с эпохи {checkpoint["epoch"]}, точность на валидации: {checkpoint["valid_accuracy"]:.2f}%')

model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f'Точность на тестовой выборке: {test_accuracy:.2f}%')