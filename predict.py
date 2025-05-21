import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2-class output
model.load_state_dict(torch.load("dyslexia_model_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

# Transform (should match training transform size)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match your training input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    label = "Dyslexic" if predicted_class == 0 else "Non-Dyslexic"
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    return label, confidence

# Example usage
if __name__ == "__main__":
    test_image = "C:\\Users\\Smruti Deshpande\\Desktop\\non.png"
    predict_image(test_image)
