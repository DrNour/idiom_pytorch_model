from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


REQUIRED_COLUMNS = [
    "source_sentence",
    "idiom",
    "meaning",
    "mt_translation",
    "label",
]


@dataclass
class LabelMaps:
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def validate_dataframe(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("The dataset contains missing values in required columns.")


def build_text(row: pd.Series) -> str:
    """Create one multilingual input string for the classifier."""
    return (
        f"Source sentence: {row['source_sentence']}\n"
        f"Idiom: {row['idiom']}\n"
        f"Intended meaning: {row['meaning']}\n"
        f"Arabic machine translation: {row['mt_translation']}"
    )


def create_label_maps(labels: List[str]) -> LabelMaps:
    unique_labels = sorted(set(labels))
    label2id = {label: index for index, label in enumerate(unique_labels)}
    id2label = {index: label for label, index in label2id.items()}
    return LabelMaps(label2id=label2id, id2label=id2label)


class IdiomTranslationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        max_length: int = 192,
    ) -> None:
        validate_dataframe(dataframe)
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        text = build_text(row)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return item
