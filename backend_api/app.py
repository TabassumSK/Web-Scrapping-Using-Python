from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Define the request body format
class Tweet(BaseModel):
    text: str

# Load the pre-trained model (this will download it on first run)
# This is the "advanced" part of your project
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Initialize the FastAPI app
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="API to predict sentiment of a tweet using twitter-roberta-base-sentiment"
)

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running."}


@app.post("/predict")
def predict_sentiment(tweet: Tweet):
    """
    Predicts the sentiment of a single tweet.
    """
    # The pipeline returns a list of dicts, e.g., [{'label': 'LABEL_1', 'score': 0.9...}]
    # LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive
    prediction = sentiment_pipeline(tweet.text)
    
    # Extract the key information
    result = prediction[0]
    label = result['label']
    score = result['score']

    # Remap labels to be more human-readable
    if label == "LABEL_0":
        sentiment = "Negative"
    elif label == "LABEL_1":
        sentiment = "Neutral"
    else:  # LABEL_2
        sentiment = "Positive"

    return {"text": tweet.text, "sentiment": sentiment, "confidence": score}