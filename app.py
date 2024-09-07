import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

analyzer = SentimentIntensityAnalyzer()

# Load model and tokenizer at startup
def load_keras_model():
    return load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    with open('models/tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

model = load_keras_model()
tokenizer = load_tokenizer()

def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("user_text")
        vader_sentiment = analyzer.polarity_scores(text)
        custom_sentiment = sentiment_analysis(text)
        
        # Combine VADER and custom model results
        sentiment = {
            "vader": vader_sentiment,
            "custom_model_positive": custom_sentiment
        }
        
        return render_template('form.html', sentiment=sentiment, text=text)
    
    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)