import os
import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Idiom Translation Error Classifier",
    page_icon="🔍",
    layout="centered"
)

st.title("English–Arabic Idiom Translation Error Classifier")

st.write(
    "This app demonstrates a PyTorch-based pipeline for classifying errors in "
    "English–Arabic idiom translation. The model can classify outputs as acceptable, "
    "literal errors, semantic errors, omissions, or cultural errors."
)

st.subheader("Try an example")

source_sentence = st.text_area(
    "English source sentence",
    "He finally spilled the beans about the plan."
)

idiom = st.text_input(
    "Idiom",
    "spill the beans"
)

meaning = st.text_input(
    "Meaning",
    "reveal a secret"
)

mt_translation = st.text_area(
    "Arabic machine translation",
    "سكب الفاصوليا أخيرا حول الخطة."
)

model_dir = "noureldin80/idiom-translation-error-model"

def simple_demo_classifier(arabic_translation):
    """
    Simple fallback demo classifier.
    This lets the Streamlit app run before the trained PyTorch model is uploaded.
    """
    literal_words = ["فاصوليا", "دلو", "قطط", "كلاب", "جليد", "يد"]
    if any(word in arabic_translation for word in literal_words):
        return "literal_error"
    return "acceptable"


if st.button("Classify Translation"):
    if True:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.eval()

            text = (
                f"Source sentence: {source_sentence}\n"
                f"Idiom: {idiom}\n"
                f"Meaning: {meaning}\n"
                f"Arabic translation: {mt_translation}"
            )

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            with torch.no_grad():
                outputs = model(**inputs)
                prediction_id = torch.argmax(outputs.logits, dim=1).item()

            predicted_label = model.config.id2label.get(
    prediction_id,
    model.config.id2label.get(str(prediction_id), str(prediction_id))
)
            st.success(f"Predicted label: {predicted_label}")

        except Exception as e:
            st.error("The trained model exists, but it could not be loaded.")
            st.code(str(e))

    else:
        predicted_label = simple_demo_classifier(mt_translation)
        st.warning(
            "No trained PyTorch model was found yet. Showing a demo prediction instead."
        )
        st.success(f"Demo predicted label: {predicted_label}")

st.subheader("Sample dataset")

data_path = "data/sample_idiom_translation_errors.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.dataframe(df)
else:
    st.info("Sample dataset file not found.")