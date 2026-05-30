# Women's Clothing E-Commerce Sentiment Analysis
### Transformer-Based NLP with RoBERTa (Hugging Face)

Sentiment analysis on 23,486 women's clothing e-commerce reviews using a pre-trained RoBERTa transformer model from Hugging Face. The project combines NLP inference with exploratory data analysis to surface patterns in customer sentiment across product categories, divisions, and departments.

---

## The Problem

E-commerce platforms generate thousands of customer reviews daily, but raw ratings alone don't capture sentiment nuance. This project asks: **can a transformer-based model reveal which product categories, divisions, and departments consistently generate positive or negative customer sentiment** — and how does that align with star ratings?

---

## Dataset

**Source:** [Kaggle — Women's Clothing E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

| Detail | Value |
|---|---|
| Rows | 23,486 reviews |
| Columns | 11 features |
| Sentiment sample | 2,000 rows (RoBERTa inference) |

**Features include:** Clothing ID, Age, Review Title, Review Text, Star Rating, Recommended IND, Positive Feedback Count, Division Name, Department Name, Class Name

> **Note:** The dataset is not included in this repository. Download it from Kaggle as a `.zip` archive and extract `Womens Clothing E-Commerce Reviews.csv` into the project root before running the notebook.

---

## Model

**`cardiffnlp/twitter-roberta-base-sentiment`** — a RoBERTa model fine-tuned on ~58 million tweets, available via Hugging Face Transformers. It classifies text into three sentiment labels: **Positive**, **Neutral**, and **Negative**.

RoBERTa (Robustly Optimized BERT Pretraining Approach) is a transformer-based model that understands context bidirectionally — meaning it reads the full sentence before assigning meaning to any word. This makes it significantly more accurate on informal, opinionated text like product reviews compared to rule-based approaches like VADER.

---

## Analysis Questions

**Q1 — Which products generate the most positive and most negative sentiment?**
Products ranked by aggregated RoBERTa sentiment scores to identify top-performing and underperforming items.

**Q2 — How does sentiment vary by Division, Department, and Class?**
Breakdown of sentiment distribution across the three product hierarchy levels — General, General Petite, Initmates divisions; Tops, Dresses, Bottoms, Intimate, Jackets, Trend departments.

**Q3 — What is the correlation between product category and sentiment score?**
Statistical relationship between categorical groupings and RoBERTa confidence scores for each sentiment label.

---

## Notebook Structure

```
1. Library Imports & Style Configuration
2. Load Dataset (Google Drive / Kaggle CSV)
3. RoBERTa Inference — 2,000 review sample
4. Exploratory Data Analysis — full dataset + sentiment columns
5. Q1: Products with highest positive and negative sentiment
6. Q2: Sentiment by Division, Department, and Class
7. Q3: Correlation between categories and sentiment scores
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/barcelonajames/womens-clothing-sentiment-analysis.git
cd womens-clothing-sentiment-analysis
```

### 2. Set up the environment
```bash
conda create -n sentiment-analysis python=3.11 -y
conda activate sentiment-analysis
pip install -r requirements.txt
```

### 3. Add the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) and place the CSV in the project root:
```
womens-clothing-sentiment-analysis/
└── Womens Clothing E-Commerce Reviews.csv   ← place here
```

### 4. Open the notebook
```bash
jupyter notebook womens_clothing_sentiment.ipynb
```

> **Note:** The RoBERTa model (~500MB) is downloaded automatically from Hugging Face on first run. An internet connection is required.

---

## Project Structure

```
womens-clothing-sentiment-analysis/
├── womens_clothing_sentiment.ipynb   ← Main notebook (run this)
├── requirements.txt                  ← Python dependencies
├── .gitignore                        ← CSV and zip excluded
├── LICENSE                           ← MIT
└── README.md
```

---

## Tools & Libraries

| Category | Libraries |
|---|---|
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| NLP / Transformers | transformers, torch (Hugging Face) |
| Model | cardiffnlp/twitter-roberta-base-sentiment |
| Environment | Python 3.11, conda |

---

## Context

Built as part of the **Uplift Code Camp Python for Data and AI Bootcamp** (2026). This project demonstrates the practical gap between traditional lexicon-based sentiment tools (VADER) and modern transformer models — RoBERTa captures sarcasm, negation, and informal language that rule-based systems consistently miss.

---

*James Aleister Barcelona — Visual Designer & Data Analyst | Davao, Philippines*
