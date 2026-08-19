"""
Fake News Detector - Streamlit App
Paste a headline/article, get a prediction, confidence score, and word-level explanation.
"""
import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

from utils import clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


def predict_proba_wrapper(texts, model, vectorizer):
    """Wrapper needed by LIME: takes raw texts, returns class probabilities."""
    cleaned = [clean_text(t) for t in texts]
    vecs = vectorizer.transform(cleaned)
    return model.predict_proba(vecs)


def main():
    st.title("📰 Fake News Detector")
    st.caption("TF-IDF + classic ML baseline, trained on the Kaggle Fake/Real News dataset")

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python train_model.py` first "
            "(after downloading the dataset into the `data/` folder)."
        )
        return

    text_input = st.text_area(
        "Paste a news headline or article:",
        height=200,
        placeholder="e.g. Scientists confirm chocolate cures all diseases...",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        check_clicked = st.button("🔍 Check News", type="primary", use_container_width=True)
    with col2:
        explain_clicked = st.button("🧠 Explain Prediction", use_container_width=True)

    if check_clicked or explain_clicked:
        if not text_input.strip():
            st.warning("Please paste some text first.")
            return

        cleaned = clean_text(text_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        label = "🟢 Real News" if pred == 1 else "🔴 Fake News"
        confidence = proba[pred] * 100

        st.subheader(label)
        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(float(proba[1]))
        st.caption(f"P(Real) = {proba[1]:.3f}  |  P(Fake) = {proba[0]:.3f}")

        if explain_clicked:
            st.divider()
            st.subheader("Why did the model decide this?")
            with st.spinner("Generating explanation..."):
                explainer = LimeTextExplainer(class_names=["Fake", "Real"])
                exp = explainer.explain_instance(
                    text_input,
                    lambda texts: predict_proba_wrapper(texts, model, vectorizer),
                    num_features=10,
                )
                html_content = exp.as_html()
                # Wrap in a light background container with larger, readable
                # text — the default LIME output is sized for a full browser
                # window, not a small embedded iframe.
                styled_html = f"""
                <div style="background-color:white; padding:16px; border-radius:8px;
                            font-size:16px; line-height:1.6; color:black;">
                    {html_content}
                </div>
                """
                components.html(styled_html, height=800, scrolling=True)

                st.caption(
                    "🟠 Orange words pushed the prediction toward **Real**, "
                    "🔵 blue words pushed it toward **Fake**. Longer bars = stronger influence."
                )

    st.divider()
    with st.expander("About this project"):
        st.markdown(
            """
            **Pipeline:** raw text → cleaning/lemmatization → TF-IDF (unigrams + bigrams) →
            classical ML classifier (best of Logistic Regression / Naive Bayes / Random Forest,
            selected automatically during training).

            **Explainability:** LIME highlights which words pushed the prediction toward
            Fake or Real, so predictions aren't a black box.

            **Dataset:** Kaggle Fake and Real News Dataset (~44k articles).
            """
        )


if __name__ == "__main__":
    main()