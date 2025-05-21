import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.title("📝 Handwritten Text Extractor (TrOCR)")

uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    # Resize image to 384x384 (recommended size for TrOCR)
    image = image.resize((384, 384))
    st.image(image, caption="Uploaded Image (resized to 384x384)", use_column_width=True)

    @st.cache(allow_output_mutation=True)
    def load_model():
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        return processor, model

    processor, model = load_model()

    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.success("✅ Extracted Text:")
        st.text_area("Extracted Text", extracted_text, height=150)

