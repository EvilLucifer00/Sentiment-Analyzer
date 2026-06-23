# Sentiment Analyzer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-success?style=for-the-badge&logo=appveyor)](https://sentiment-analyzer-bvlazmkmvswfgmkeaiskcq.streamlit.app/)

A sentiment analysis system built using pretrained DistilBERT on 
IMDB movie reviews. Achieves 89% accuracy with zero model training 
— demonstrating the power of transfer learning for NLP tasks.

## Problem
Manually analyzing thousands of product/movie reviews is 
time-consuming and inconsistent. This system automates sentiment 
classification at scale with high confidence scores.

## Model
- **Model:** distilbert-base-uncased-finetuned-sst-2-english
- **Source:** HuggingFace Transformers
- **Approach:** Zero-shot inference — no fine-tuning required
- **Dataset:** IMDB 50K Movie Reviews

## Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | 89% |
| NEGATIVE Precision | 0.86 |
| NEGATIVE Recall | 0.97 |
| NEGATIVE F1 | 0.91 |
| POSITIVE Precision | 0.94 |
| POSITIVE Recall | 0.79 |
| POSITIVE F1 | 0.86 |

## Key Findings

**1. Zero training, strong results**
DistilBERT achieves 89% accuracy straight out of the box — no 
fine-tuning, no labelled training data needed. This is the power 
of transfer learning.

**2. Model detects negative sentiment more reliably than positive**
NEGATIVE recall (0.97) significantly outperforms POSITIVE recall 
(0.79). Root cause: sarcastic or ironic positive reviews use 
logically negative words to express positive meaning — the model 
reads words but misses the deeper contextual intent.

**3. Data cleaning had minimal impact**
Removing HTML tags (`<br/>`) from reviews changed accuracy by only 
1% (89% → 88%). DistilBERT's tokenizer handles noisy input 
reasonably well — aggressive preprocessing is not always necessary.

## What I Would Improve
- Fine-tune DistilBERT on domain-specific data to improve 
  POSITIVE recall
- Add bulk processing endpoint for 1000+ reviews
- Deploy as FastAPI backend with confidence score thresholding
- Add explainability — highlight which words drove the prediction

## How to Run

```bash
pip install transformers torch pandas scikit-learn
```

Open `sentiment_analyzer.ipynb` and run all cells.

## Tech Stack
- Python
- HuggingFace Transformers
- DistilBERT
- Pandas
- scikit-learn

## Author
[Your Name] — Part of a 60-day AI Engineer roadmap
