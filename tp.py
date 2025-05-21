import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 output classes
model.load_state_dict(torch.load("dyslexia_model_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

# Define transform (must match training!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    label = "Dyslexic" if predicted_class == 0 else "Non-Dyslexic"
    return label, confidence

# Streamlit App
st.set_page_config(page_title="🧠 Dyslexia Detection (Image)", layout="centered")
st.title("📷 Handwriting-based Dyslexia Detection")
st.markdown("Upload a handwriting image to predict whether it indicates Dyslexia.")

uploaded_file = st.file_uploader("Upload Handwriting Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Predict Dyslexia"):
        label, confidence = predict_image(image)
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence Score: {confidence:.2f}")
        if label == "Dyslexic":
            st.error("⚠️ Signs of Dyslexia detected.")
        else:
            st.success("✅ No signs of Dyslexia detected.")
