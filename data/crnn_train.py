import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from crnn_model import CRNN
from torch.nn import CTCLoss
import sys

# --- Configuration ---
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
imgH, imgW = 32, 100
nclass = len(alphabet) + 1
nh = 256
batch_size = 8
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Label Converter ---
class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet + "-"
        self.char2idx = {char: i for i, char in enumerate(self.alphabet)}
        self.idx2char = {i: char for i, char in enumerate(self.alphabet)}

    def encode(self, text):
        return [self.char2idx[char] for char in text]

    def decode(self, preds):
        preds = preds.argmax(2)
        preds = preds.permute(1, 0)
        texts = []
        for pred in preds:
            char_list = []
            for i in range(len(pred)):
                if pred[i] != nclass - 1 and (i == 0 or pred[i] != pred[i - 1]):
                    char_list.append(self.idx2char[pred[i].item()])
            texts.append("".join(char_list))
        return texts

converter = LabelConverter(alphabet)

# --- Auto-generate train_labels.txt with validation ---
os.makedirs("data", exist_ok=True)
sample_count = 0

with open("data/train_labels.txt", "w") as f:
    for category in ["dyslexic", "non_dyslexic"]:
        folder = os.path.join("data", category)
        if not os.path.exists(folder):
            print(f"⚠️  Warning: Folder not found: {folder}")
            continue
        image_files = [img for img in os.listdir(folder) if img.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            print(f"⚠️  Warning: No valid images in {folder}")
        for img_name in image_files:
            label = "dys" if category == "dyslexic" else "non"
            f.write(f"{category}/{img_name},{label}\n")
            sample_count += 1

if sample_count == 0:
    print("❌ Error: No valid training images found in 'data/dyslexia' or 'data/non-dyslexia'.")
    sys.exit(1)
else:
    print(f"✅ {sample_count} training samples found and listed in train_labels.txt.")

# --- Dataset ---
class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, label_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                path, label = line.strip().split(",", 1)
                full_path = os.path.join(self.img_folder, path)
                if os.path.exists(full_path):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(os.path.join(self.img_folder, img_path)).convert("L")
        if self.transform:
            image = self.transform(image)
        label_idx = torch.tensor(converter.encode(label), dtype=torch.long)
        return image, label_idx, len(label_idx)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((imgH, imgW)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- DataLoader ---
train_dataset = OCRDataset("data", "data/train_labels.txt", transform)

if len(train_dataset) == 0:
    print("❌ Error: Dataset is empty even after processing. Please verify image formats and paths.")
    sys.exit(1)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

# --- Model ---
model = CRNN(imgH, 1, nclass, nh).to(device)
criterion = CTCLoss(blank=nclass - 1, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        images, labels, label_lengths = zip(*batch)
        images = torch.stack(images).to(device)
        targets = torch.cat(labels).to(device)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long).to(device)

        preds = model(images)  # [T, B, C]
        preds = preds.log_softmax(2)

        input_lengths = torch.full(
            size=(preds.size(1),),  # batch size
            fill_value=preds.size(0),  # actual time steps
            dtype=torch.long
        ).to(device)

        loss = criterion(preds, targets, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# --- Save model ---
torch.save(model.state_dict(), "crnn_ocr.pth")
print("✅ CRNN model saved as crnn_ocr.pth")
