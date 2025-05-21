import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np


# === STEP 1: Prepare Data Split ===


source_dys = './data/raw/dyslexic'
source_non = './data/raw/non_dyslexic'


train_dir = './data/split_data/train'
val_dir = './data/split_data/val'


os.makedirs(f'{train_dir}/dyslexic', exist_ok=True)
os.makedirs(f'{train_dir}/non_dyslexic', exist_ok=True)
os.makedirs(f'{val_dir}/dyslexic', exist_ok=True)
os.makedirs(f'{val_dir}/non_dyslexic', exist_ok=True)


def split_and_copy(source, label):
    images = [img for img in os.listdir(source) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    for img in train_imgs:
        shutil.copy(os.path.join(source, img), os.path.join(train_dir, label, img))
    for img in val_imgs:
        shutil.copy(os.path.join(source, img), os.path.join(val_dir, label, img))


if not os.listdir(f"{train_dir}/dyslexic"):
    split_and_copy(source_dys, 'dyslexic')
    split_and_copy(source_non, 'non_dyslexic')


# === STEP 2: Training Configuration ===


batch_size = 8
epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)


# Print class distributions
print("Train class distribution:", Counter(train_dataset.targets))
print("Validation class distribution:", Counter(val_dataset.targets))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Model setup
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)


# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)


# === STEP 3: Training Loop ===
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0


    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}%")


    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    all_preds = []
    all_labels = []


    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    scheduler.step(avg_val_loss)


# Save model
torch.save(model.state_dict(), 'dyslexia_model_resnet18.pth')
print("✅ Model saved as dyslexia_model_resnet18.pth")


# === STEP 4: Evaluation ===
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))


print("\n📉 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


