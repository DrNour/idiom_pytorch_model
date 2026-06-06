# Idiom Translation Error Classifier

A GitHub-ready PyTorch project for studying **English-Arabic idiom translation**. The model classifies machine-translated Arabic outputs into error/quality categories such as `acceptable`, `literal_error`, `omission`, and `semantic_error`.

This is designed for a research paper on idiom translation, machine translation, and post-editing effort.

## Research use case

Given:

- an English source sentence containing an idiom,
- the idiom itself,
- the intended idiomatic meaning,
- an Arabic machine translation output,

predict whether the translation is acceptable or problematic.

Example:

```text
Source: He finally spilled the beans about the plan.
Idiom: spill the beans
Meaning: reveal a secret
MT output: سكب الفاصوليا حول الخطة أخيرا.
Label: literal_error
```

## Project structure

```text
idiom_pytorch_model/
├── data/
│   └── sample_idiom_translation_errors.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
├── scripts/
│   └── run_train.sh
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python -m src.train \
  --data_path data/sample_idiom_translation_errors.csv \
  --model_name xlm-roberta-base \
  --output_dir outputs/idiom_error_model \
  --epochs 3 \
  --batch_size 4 \
  --max_length 192
```

For a lighter model, try:

```bash
python -m src.train --model_name distilbert-base-multilingual-cased
```

## Make a prediction

After training:

```bash
python -m src.predict \
  --model_dir outputs/idiom_error_model \
  --source_sentence "He finally spilled the beans about the plan." \
  --idiom "spill the beans" \
  --meaning "reveal a secret" \
  --mt_translation "سكب الفاصوليا حول الخطة أخيرا."
```

## Dataset format

Your CSV should contain these columns:

```text
source_sentence,idiom,meaning,mt_translation,label
```

Recommended labels:

- `acceptable`
- `paraphrase_acceptable`
- `literal_error`
- `omission`
- `semantic_error`
- `cultural_error`

## Suggested paper framing

Possible title:

**From Literal Error to Post-Editing Effort: A PyTorch-Based Study of English-Arabic Idiom Translation**

Possible research questions:

1. Can a transformer-based PyTorch classifier identify literal translation errors in English-Arabic idiom translation?
2. Which idiom types produce the highest proportion of literal or semantic errors?
3. Can model predictions be used to estimate post-editing difficulty?

## GitHub notes

Do not commit trained model weights unless necessary. The `.gitignore` file excludes the `outputs/` directory by default.

To push to GitHub:

```bash
git init
git add .
git commit -m "Initial idiom translation classifier"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/idiom-pytorch-model.git
git push -u origin main
```
