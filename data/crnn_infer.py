import os
import torch
from PIL import Image
from torchvision import transforms
from crnn_model import CRNN
import torch.nn.functional as F

# Configs (same as training)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
imgH, imgW = 32, 100
nclass = len(alphabet) + 1
nh = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label Converter class (same as training)
class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet + "-"
        self.char2idx = {char: i for i, char in enumerate(self.alphabet)}
        self.idx2char = {i: char for i, char in enumerate(self.alphabet)}

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

# Simple decode function just for debugging (no blank removal)
def simple_decode(preds):
    preds_idx = preds.argmax(2)
    preds_idx = preds_idx.permute(1, 0)
    texts = []
    for pred in preds_idx:
        text = ''.join([alphabet[i] if i < len(alphabet) else '' for i in pred])
        texts.append(text)
    return texts

# Image preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.Resize((imgH, imgW)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the trained CRNN model
model = CRNN(imgH, 1, nclass, nh).to(device)
state_dict = torch.load("crnn_ocr.pth", map_location=device)
print("Loaded model keys:", list(state_dict.keys()))
model.load_state_dict(state_dict)
model.eval()

def ocr_predict(image_path):
    image = Image.open(image_path).convert("L")
    print("Original image size:", image.size)
    image = transform(image).unsqueeze(0).to(device)  # add batch dim
    print("Tensor shape after transform:", image.shape)

    with torch.no_grad():
        preds = model(image)
    print("Raw preds shape:", preds.shape)
    print("Raw preds sample (first 5 classes for first time step):", preds[:, 0, :5])

    # Try both decodings
    pred_text = converter.decode(preds)[0]
    simple_text = simple_decode(preds)[0]
    probs = F.softmax(preds, dim=2)
    max_probs, max_indices = torch.max(probs, dim=2)
    print("Max probs shape:", max_probs.shape)
    print("Max probs sample:", max_probs[:, 0])
    print("Max indices sample:", max_indices[:, 0])

    print("Decoded text:", pred_text)
    print("Simple decoded text:", simple_text)
    return pred_text

if __name__ == "__main__":
    test_img_path = "data/non_dyslexic/14.jpg"  # change to your test image path
    text = ocr_predict(test_img_path)
    print(f"Recognized Text: {text}")
