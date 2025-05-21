import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import io

st.title("📝 OCR + Correction Debug")

@st.cache_resource
def load_models():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    grammar_tokenizer = AutoTokenizer.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
    grammar_model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
    return processor, ocr_model, grammar_tokenizer, grammar_model

processor, ocr_model, grammar_tokenizer, grammar_model = load_models()

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    if st.button("Extract and Correct"):
        # OCR
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Grammar Correction
        prompt = "Correct this to proper English: " + ocr_text
        inputs = grammar_tokenizer(prompt, return_tensors="pt")
        outputs = grammar_model.generate(inputs.input_ids, max_length=256)
        corrected = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("🔍 OCR Output")
        st.code(ocr_text)

        st.subheader("✅ Corrected Text")
        st.success(corrected)

