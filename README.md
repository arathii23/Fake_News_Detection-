# Fake News Detection: BERT-based Classification with Streamlit

[View Live Application](https://fakenewsdetection-9ix5r3mbvwd9vakvdh6j3f.streamlit.app/)

## Overview
This project implements a robust fake news detection system leveraging the BERT (Bidirectional Encoder Representations from Transformers) model.

While traditional machine learning approaches often rely on frequency-based features, this project utilizes BERT to incorporate contextual understanding of natural language. This significantly improves the model's ability to distinguish between reliable and misleading news articles based on subtle linguistic patterns.

## Methodology
Traditional methods (e.g., Bag of Words, TF-IDF) frequently overlook the semantic context of words. To address this, we fine-tuned the `bert-base-uncased` model using the Hugging Face Transformers library. BERT's attention mechanism allows it to interpret the meaning of words relative to their surrounding text, making it highly effective for misinformation detection.

### Dataset
The model was trained on the "Fake and Real News Dataset" sourced from Kaggle.
* **Source:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## Model Performance
The fine-tuned BERT model achieved the following performance metric on the validation dataset:

* **F1 Score:** 99.98%

## Streamlit Interface
A lightweight Streamlit web application was developed to demonstrate the model's inference capabilities in real-time.

### Key Features
1.  **Input:** Users can input a news headline or full article text.
2.  **Inference:** The model processes the input via the fine-tuned BERT architecture.
3.  **Classification:** The application outputs a binary classification:
    * Real News
    * Fake News

## Project Structure

```text
├── artifacts/             # Stored model weights and tokenizer files
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and Model Training
├── src/                   # Source code for preprocessing and inference
├── app.py                 # Streamlit application entry point
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
