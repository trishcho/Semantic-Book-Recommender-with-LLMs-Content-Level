# Semantic Book Recommender with LLMs — Content Level

Semantic book recommender powered by LLMs: OpenAI embeddings + Chroma vector search to surface similar titles from the 7k Books dataset.
Zero-shot Fiction/Nonfiction tagging and emotion-based re-ranking, wrapped in a clean Gradio UI for instant recommendations.


<img width="1459" height="668" alt="Image" src="https://github.com/user-attachments/assets/99c5877e-ead4-4a7b-a134-b92fe1cc2ae0" />

Kaggle dataset (7k books): https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata?resource=download

This repo shows how to build a semantic book recommender using modern LLM tooling:

* Vector search over book descriptions (Chroma + OpenAI embeddings)

* Zero-shot text classification (e.g., Fiction vs Nonfiction) with 🤗 Transformers

* Emotion/tone scoring (optionally) to re-rank results

* Gradio dashboard for interactive recommendations

### What’s inside

* data-exploration.ipynb — light cleaning and prep of book metadata/descriptions

* vector-search.ipynb — build embeddings + Chroma vector store

* text-classification.ipynb — zero-shot classification with facebook/bart-large-mnli

* sentiment-analysis.ipynb — optional emotion features (joy, fear, sadness, etc.)

* gradio-dashboard.py — the UI: query, filter (category/tone), show covers & captions

pip install -U pip
pip install \
  gradio pandas numpy python-dotenv \
  transformers torch \
  langchain langchain-community langchain-chroma chromadb \
  langchain-openai langchain-text-splitters


### Add your OpenAI key (for embeddings)
Create a .env in the project root:

OPENAI_API_KEY=sk-...

### Get the data

Download the CSV from Kaggle:
https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata?resource=download

Put it in the project folder (or update code paths accordingly).

### (Optional) Build a line-delimited text file for descriptions

import pandas as pd, pathlib
books = pd.read_csv("books_with_emotions.csv")  # or the Kaggle CSV you preprocessed
pathlib.Path("tagged_description.txt").write_text(
    "\n".join(books["tagged_description"].dropna().astype(str)),
    encoding="utf-8"
)


### Run the dashboard
python gradio-dashboard.py
### then open the printed URL (e.g., http://127.0.0.1:7860)



### Notes & tips

from langchain_chroma import Chroma

### Community bundle
from langchain_community.vectorstores import Chroma

### Text splitters

CharacterTextSplitter requires a positive chunk_size and non-negative chunk_overlap.

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=500,   # > 0
  chunk_overlap=50  # >= 0
)

### pandas read_csv

The old error_bad_lines/warn_bad_lines args were removed. Use on_bad_lines="skip" instead.

import pandas as pd

df = pd.read_csv(
  "file.csv",
  sep=";",
  on_bad_lines="skip",  # replaces error_bad_lines/warn_bad_lines
  encoding="latin-1",
  engine="python"       # helpful for irregular CSVs
)


### Torch on macOS (M-series)

Check Apple GPU (MPS) and pass an int device index to 🤗 pipelines (0 = GPU/MPS, -1 = CPU). Do not pass "mps" as a string.

import torch
from transformers import pipeline

print("MPS available:", torch.backends.mps.is_available())
device = 0 if torch.backends.mps.is_available() else -1

pipe = pipeline(
  "zero-shot-classification",
  model="facebook/bart-large-mnli",
  device=device
)

