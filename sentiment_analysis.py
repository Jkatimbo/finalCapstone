# sentiment_analysis.spacy

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

# Step 1: Load and preprocess the dataset
data = pd.read_csv('amazon_product_reviews.csv')
reviews_data = data['review.text']
clean_data = data.dropna(subset=['review.text'])

# Step 2: Define function for sentiment analysis
def analyze_sentiment(review):
    doc = nlp(review)
    # Use the .sentiment attribute to get the sentiment
    sentiment = doc._.sentiment
    return sentiment

# Function for overall sentiment analysis
def overall_sentiment_analysis(reviews):
    sentiments = [analyze_sentiment(review) for review in reviews]
    return sentiments

# Step 3: Test the sentiment analysis function on sample reviews
sample_reviews = ["This product is amazing!", "Not satisfied with the quality."]
for review in sample_reviews:
    sentiment_result = analyze_sentiment(review)
    print(f"Sentiment of '{review}': {sentiment_result}")

