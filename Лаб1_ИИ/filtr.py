import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import random
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # для прогресс-бара

# Сбор данных
train_dataset = []
test_dataset = []
test_target = []

for idx, (root, dirs, files) in enumerate(os.walk("archive/simpsons_dataset")):
    for file in files:
        train_dataset.append(os.path.join(root, file))

for idx, (root, dirs, files) in enumerate(os.walk("archive/kaggle_simpson_testset/kaggle_simpson_testset")):
    for file in files:
        test_dataset.append(os.path.join(root, file))

for path in test_dataset:
    name = os.path.basename(path).split('_')
    test_target.append('_'.join(name[:-1]))

# Модель
model = torchvision.models.resnet18(weights='DEFAULT')
layer4_features = None

def get_features(module, inputs, output):
    global layer4_features
    layer4_features = output

model.layer4.register_forward_hook(get_features)
model.eval()

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === ОПТИМИЗАЦИЯ 1: Предварительное вычисление эмбеддингов для train ===
print("Вычисление эмбеддингов для тренировочных изображений...")
train_embeddings = {}  # словарь: путь -> эмбеддинг

with torch.no_grad():
    for img_path in tqdm(train_dataset):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Получаем имя персонажа из пути
        char_name = os.path.basename(os.path.dirname(img_path))
        
        # Преобразуем и получаем эмбеддинг
        img_tensor = preprocess(img)
        model(img_tensor[None, :, :, :])
        embedding = layer4_features.mean(dim=[2, 3]).squeeze()
        
        # Сохраняем
        if char_name not in train_embeddings:
            train_embeddings[char_name] = []
        train_embeddings[char_name].append(embedding.cpu())

# === ОПТИМИЗАЦИЯ 2: Использование batch processing ===
norm_dataset = []

with torch.no_grad():
    for idx, img_test in enumerate(tqdm(test_dataset)):
        print(idx)
        similar = 0
        
        # Получаем эмбеддинг тестового изображения
        img_testi = cv2.imread(img_test)
        if img_testi is None:
            continue
        img_testi = cv2.cvtColor(img_testi, cv2.COLOR_BGR2RGB)
        
        testi = preprocess(img_testi)
        model(testi[None, :, :, :])
        test_embedding = layer4_features.mean(dim=[2, 3]).squeeze().cpu()
        
        # Сравниваем с предвычисленными эмбеддингами того же персонажа
        target_char = test_target[idx]
        if target_char in train_embeddings:
            train_embs = train_embeddings[target_char]
            
            # === ОПТИМИЗАЦИЯ 3: Векторизованное вычисление косинусного сходства ===
            test_embedding_norm = test_embedding / test_embedding.norm()
            
            for train_emb in train_embs:
                # Нормализованное скалярное произведение = косинусное сходство
                cos_sim = torch.dot(test_embedding_norm, train_emb / train_emb.norm())
                probability = (cos_sim.item() + 1) / 2
                
                if probability > 0.8:
                    similar += 1
        
        if similar == 0:
            norm_dataset.append(img_test)

print(norm_dataset)