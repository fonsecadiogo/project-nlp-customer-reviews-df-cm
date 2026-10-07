# Streamlit MVP for customer-review sentiment analysis.

import joblib
import streamlit as st
from pathlib import Path


# Find the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the paths to the saved model files.
TFIDF_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_svm_balanced.joblib"


# Load the trained TF-IDF vectorizer.
tfidf = joblib.load(TFIDF_PATH)

# Load the trained Balanced Linear SVM model.
svm_balanced = joblib.load(MODEL_PATH)


# Configure the Streamlit page.
st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="💬",
)


# App title.
st.title("Customer Review Sentiment Analyzer")

# Short explanation.
st.write(
    "Enter a customer review and the model will classify it "
    "as Positive, Neutral, or Negative."
)


# Text area where the user enters a review.
review = st.text_area(
    "Customer review",
    placeholder="Example: This product is easy to use and works perfectly."
)


# Run the prediction when the button is clicked.
if st.button("Analyze Sentiment"):

    # Remove unnecessary spaces at the beginning and end.
    cleaned_review = review.strip()

    # Prevent empty input.
    if not cleaned_review:
        st.warning("Please enter a customer review before analyzing.")

    else:
        # Convert the review into TF-IDF numerical features.
        review_tfidf = tfidf.transform([cleaned_review])

        # Predict the sentiment with the Balanced Linear SVM.
        prediction = svm_balanced.predict(review_tfidf)[0]

        # Display the result.
        st.subheader("Predicted Sentiment")

        if prediction == "positive":
            st.success("Positive")

        elif prediction == "neutral":
            st.info("Neutral")

        else:
            st.error("Negative")