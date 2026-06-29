# Sentiment Analysis with DistilBERT — Multidominio

Binary sentiment classifier (positive / negative) fine-tuned on DistilBERT using reviews from four different domains.

## Model & Demo

- 🤗 Model: [Drocho/distilbert-sentiment](https://huggingface.co/Drocho/distilbert-sentiment)
- 🚀 Demo: [Drocho/distilbert-sentiment-demo](https://huggingface.co/spaces/Drocho/distilbert-sentiment-demo)

## Dataset

Four public datasets were combined and unified into a single binary sentiment dataset:

| Source | Domain |
|---|---|
| IMDb | Movie reviews |
| Amazon Polarity | Product reviews |
| Yelp Review Full | Restaurant & local business reviews |
| SST-2 | Short movie review fragments |

The final dataset `Drocho/sentiment-unified` contains **36,496 samples** split 80/10/10:
- Train: 29,196 samples
- Validation: 3,650 samples
- Test: 3,650 samples

All splits are balanced between positive and negative labels.

## Pipeline

```
1. Data acquisition     → Load datasets from Hugging Face Hub
2. Cleaning             → Remove HTML tags, URLs, special characters, lowercase
3. Balancing            → Stratified sampling, max 10,000 samples per source
4. Deduplication        → MD5 hash-based deduplication
5. Splitting            → 80 / 10 / 10 train / val / test
6. Tokenization         → WordPiece, max 512 tokens, padding + truncation
7. Fine-tuning          → DistilBERT + classification head, AdamW, 3 epochs
8. Evaluation           → Accuracy, F1-score, Precision, Recall vs TF-IDF baseline
```

## Model

Base model: `distilbert-base-uncased` (66M parameters, 6 transformer layers).

A classification head was added on top of the `[CLS]` token output:
- Linear pre-classification layer
- ReLU activation
- Dropout (0.2)
- Linear output layer → 2 classes

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 512 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Early stopping patience | 2 epochs |
| Seed | 42 |

## Results

| Metric | DistilBERT | TF-IDF + LR (baseline) |
|---|---|---|
| Accuracy | **92.85%** | 88.22% |
| F1-score | **92.63%** | 87.90% |
| Precision | **93.61%** | 88.55% |
| Recall | **91.68%** | 87.26% |

Training completed in ~275 seconds on a NVIDIA GeForce RTX 5070 Ti.

## Repository Structure

```
├── scripts/
│   └── build_dataset.py       # Dataset construction and preprocessing
├── notebooks/
│   └── finetuning_distilbert.ipynb  # Fine-tuning, evaluation and results
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```
