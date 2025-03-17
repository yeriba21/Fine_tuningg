# -*- coding: utf-8 -*-
"""Projet Fine-Tuning avec PyTorch"""

# Installation des dépendances nécessaires (à exécuter dans le terminal)
# pip install datasets torch torchvision numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from datasets import load_dataset
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Configuration
BATCH_SIZE = 64  # Taille du batch
NUM_EPOCHS = 5   # Nombre d'époques
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du dataset Fruits-360
dataset = load_dataset("PedroSampaio/fruits-360")

# Définition des transformations
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionnement des images
    transforms.RandomHorizontalFlip(),  # Augmentation des données
    transforms.ToTensor(),  # Conversion en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionnement des images
    transforms.ToTensor(),  # Conversion en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
])

# Classe personnalisée pour gérer les données
class FruitsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Création des datasets et dataloaders
train_dataset = FruitsDataset(dataset['train'], transform=train_transform)
test_dataset = FruitsDataset(dataset['test'], transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # Chargement des données
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Chargement du modèle pré-entraîné (ResNet-50)
model = models.resnet50(pretrained=True)
num_classes = len(set(dataset['train']['label']))
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adaptation de la dernière couche
model.to(DEVICE)  # Déplacement du modèle sur le GPU (si disponible)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()  # Fonction de perte
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Optimiseur Adam

# Mixed Precision Training
scaler = GradScaler()

# Boucle d'entraînement
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Mixed Precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

# Sauvegarde du modèle
torch.save(model.state_dict(), 'model.pth')
print("Modèle sauvegardé sous 'model.pth'")

# Évaluation sur le dataset de test
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")