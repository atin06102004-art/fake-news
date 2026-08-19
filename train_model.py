"""
Train a Fake News Detection model.

Expected data (Kaggle "Fake and Real News Dataset"):
    data/Fake.csv   -> columns: title, text, subject, date
    data/True.csv   -> columns: title, text, subject, date
Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Usage:
    python train_model.py
"""
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from utils import clean_text


def strip_source_tags(text: str) -> str:
    """
    Removes wire-service datelines like 'WASHINGTON (Reuters) - ' from the
    start of articles. Without this, the model can learn to spot the tag
    itself instead of learning real vs. fake writing patterns, since almost
    all 'True.csv' articles carry one and almost no 'Fake.csv' articles do.
    This is a known leakage issue with this specific Kaggle dataset.
    """
    if not isinstance(text, str):
        return ""
    # Matches patterns like "CITY (Reuters) - " or "CITY, STATE (Reuters) - "
    text = re.sub(r"^[A-Z][A-Za-z,.\s]{0,40}\(Reuters\)\s*-\s*", "", text)
    return text


def load_data():
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake["label"] = 0  # 0 = fake
    real["label"] = 1  # 1 = real

    fake["text"] = fake["text"].apply(strip_source_tags)
    real["text"] = real["text"].apply(strip_source_tags)

    df = pd.concat([fake, real], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    df = df[["content", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning text (this can take a few minutes on the full dataset)...")
    df["clean_content"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = {}
    best_model_name, best_model, best_acc = None, None, 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"{name} accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

        if acc > best_acc:
            best_model_name, best_model, best_acc = name, model, acc

    print("\n=== Summary ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    print(f"\nBest model: {best_model_name} ({best_acc:.4f})")

    # Save best model + vectorizer for the Streamlit app
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("\nSaved models/best_model.pkl and models/vectorizer.pkl")


if __name__ == "__main__":
    main()