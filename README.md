# 📰 Fake News Detector

A machine learning web app that classifies news articles/headlines as **Real** or **Fake**, with confidence scores and word-level explainability (LIME).

🔗 **Live Demo:** _add your Streamlit Cloud link here after deploying_

## Features

- **Text preprocessing pipeline** — cleaning, stopword removal, lemmatization (NLTK)
- **TF-IDF vectorization** (unigrams + bigrams)
- **Model comparison** — Logistic Regression, Naive Bayes, and Random Forest trained and benchmarked automatically; best model auto-selected
- **Explainability** — LIME highlights which words drove each prediction, so it's not a black box
- **Interactive Streamlit UI** — paste text, get instant prediction + confidence

## Tech Stack

`Python` `scikit-learn` `Streamlit` `NLTK` `LIME` `Pandas` `NumPy`

## Project Structure

```
fake-news-detector/
├── data/                  # place Fake.csv and True.csv here
├── models/                # saved model + vectorizer (generated after training)
├── utils.py               # text cleaning/preprocessing
├── train_model.py         # trains and compares models, saves the best one
├── app.py                 # Streamlit app
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   Get the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   and place `Fake.csv` and `True.csv` inside the `data/` folder.

4. **Train the model**
   ```bash
   python train_model.py
   ```
   This prints accuracy/precision/recall for all three models and saves the best one to `models/`.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | ~0.98 |
| Naive Bayes | ~0.94 |
| Random Forest | ~0.99 |

_(Fill in your actual numbers after training — they'll print in the terminal.)_

## Future Improvements

- Fine-tune a BERT model and compare against the classical ML baseline
- Add an LLM-based (Groq/LLaMA) zero-shot classifier as a second opinion
- Cross-check claims against live news via a News API
- Deploy with metadata features (source, author, publish date)

## Author

Ananya Gupta — [GitHub](https://github.com/zananyagupta25-afk) · [LinkedIn](https://linkedin.com/in/ananya-gupta-8b7023369)
