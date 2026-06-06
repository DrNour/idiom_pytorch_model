from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_text(source_sentence: str, idiom: str, meaning: str, mt_translation: str) -> str:
    return (
        f"Source sentence: {source_sentence}\n"
        f"Idiom: {idiom}\n"
        f"Intended meaning: {meaning}\n"
        f"Arabic machine translation: {mt_translation}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict idiom translation error type.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--source_sentence", type=str, required=True)
    parser.add_argument("--idiom", type=str, required=True)
    parser.add_argument("--meaning", type=str, required=True)
    parser.add_argument("--mt_translation", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "training_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        id2label = {int(key): value for key, value in metadata["id2label"].items()}
        max_length = args.max_length or metadata.get("max_length", 192)
    else:
        id2label = None
        max_length = args.max_length or 192

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    text = build_text(args.source_sentence, args.idiom, args.meaning, args.mt_translation)
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        predicted_id = int(torch.argmax(probabilities).item())

    predicted_label = id2label[predicted_id] if id2label else str(predicted_id)

    print(f"Predicted label: {predicted_label}")
    print("Probabilities:")
    for index, probability in enumerate(probabilities.tolist()):
        label = id2label[index] if id2label else str(index)
        print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
