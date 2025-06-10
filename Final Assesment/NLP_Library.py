from textblob import TextBlob

def get_sentiment(sentence):
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity # ranges from -1 (negaative) to 1 (positive)

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment

# Example
example = "I feel neither good nor bad about the results of the race"
print(f"Sentiment: {get_sentiment(example)}")