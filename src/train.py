from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset import IdiomTranslationDataset, create_label_maps, validate_dataframe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, dataloader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    predictions = []
    gold_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch_to_device(batch, device)
            outputs = model(**batch)
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(batch_predictions.cpu().tolist())
            gold_labels.extend(batch["labels"].cpu().tolist())

    return {
        "accuracy": accuracy_score(gold_labels, predictions),
        "macro_f1": f1_score(gold_labels, predictions, average="macro", zero_division=0),
        "predictions": predictions,
        "gold_labels": gold_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch idiom translation error classifier.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV dataset.")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Hugging Face model name.")
    parser.add_argument("--output_dir", type=str, default="outputs/idiom_error_model", help="Where to save the trained model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    validate_dataframe(df)

    label_maps = create_label_maps(df["label"].tolist())
    num_labels = len(label_maps.label2id)

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={str(key): value for key, value in label_maps.id2label.items()},
        label2id=label_maps.label2id,
    )

    train_dataset = IdiomTranslationDataset(train_df, tokenizer, label_maps.label2id, args.max_length)
    test_dataset = IdiomTranslationDataset(test_df, tokenizer, label_maps.label2id, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    print(f"Training on device: {device}")
    print(f"Labels: {label_maps.label2id}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            batch = batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        average_loss = epoch_loss / max(len(train_loader), 1)
        metrics = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch + 1}: "
            f"loss={average_loss:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    final_metrics = evaluate(model, test_loader, device)
    print("\nClassification report:")
    print(
        classification_report(
            final_metrics["gold_labels"],
            final_metrics["predictions"],
            target_names=[label_maps.id2label[index] for index in range(num_labels)],
            zero_division=0,
        )
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "label2id": label_maps.label2id,
        "id2label": {str(key): value for key, value in label_maps.id2label.items()},
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
    }

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"\nSaved model to: {output_dir}")


if __name__ == "__main__":
    main()
