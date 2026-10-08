# Load the saved model files and test them outside the notebook.

import joblib
from pathlib import Path


# Move from app/test_sentiment_model.py to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the paths to the saved model files.
TFIDF_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_svm_balanced.joblib"


# Load the trained TF-IDF vectorizer.
tfidf = joblib.load(TFIDF_PATH)

# Load the trained Balanced Linear SVM model.
svm_balanced = joblib.load(MODEL_PATH)


# Example review to test the complete prediction pipeline.
review = "This product is very easy to use and works perfectly."


# Convert the review text into TF-IDF numerical features.
review_tfidf = tfidf.transform([review])


# Predict the sentiment.
prediction = svm_balanced.predict(review_tfidf)


# Show the result.
print("Review:")
print(review)

print("\nPredicted sentiment:")
print(prediction[0])
