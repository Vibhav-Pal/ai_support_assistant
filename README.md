# AI Support Assistant

A lightweight NLP tool combining intent classification, text summarization,
and content-based recommendations — built with Hugging Face Transformers,
scikit-learn, and Streamlit.

## Features
- Zero-shot intent classification (no training data required)
- Abstractive text summarization (BART)
- TF-IDF based content recommendation engine
- Simple Streamlit web interface tying all three together

## Tech Stack
Python, Hugging Face Transformers, scikit-learn, Streamlit, Pandas

## Project Structure
```
ai-support-assistant/
├── app.py              # Streamlit UI, combines all 3 modules
├── classifier.py       # Zero-shot intent classification
├── summarizer.py       # Abstractive text summarization
├── recommender.py      # TF-IDF based FAQ recommender
├── data/
│   └── faqs.csv         # Sample FAQ dataset
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Why these design choices
Pretrained models were used for classification and summarization to prioritize
speed and reliability within a short timeline, while still demonstrating a
practical understanding of NLP pipelines: classification, generation, and
retrieval/similarity-based recommendation.

## Known limitation (and what I learned from it)
The TF-IDF recommender relies on exact word overlap, so semantically similar
queries with different wording (e.g. "freezing" vs "lagging") aren't always
matched correctly. This is a well-known limitation of bag-of-words methods —
a production system would use semantic embeddings (e.g. sentence-transformers)
to capture meaning rather than just word overlap. See the "Future Improvements"
section below for the planned fix.

## Future Improvements
- Replace TF-IDF recommender with semantic search using sentence embeddings
- Fine-tune a custom intent classifier on labeled support ticket data instead
  of relying solely on zero-shot classification
- Add conversation memory for multi-turn support conversations
- Scale the recommender with a vector database (e.g. FAISS) for larger FAQ sets

## Live Demo

