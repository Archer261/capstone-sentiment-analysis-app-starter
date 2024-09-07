import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask application
app = Flask(__name__)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load model and tokenizer at startup
def load_keras_model():
    """
    Load the pre-trained Keras model for sentiment analysis.
    
    Returns:
    - Keras model object
    """
    return load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    """
    Load the pre-trained tokenizer for text preprocessing.
    
    Returns:
    - Keras Tokenizer object
    """
    with open('models/tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

# Load model and tokenizer globally
model = load_keras_model()
tokenizer = load_tokenizer()

def sentiment_analysis(input):
    """
    Perform sentiment analysis using the custom trained model.
    
    Args:
    - input (str): The text to analyze
    
    Returns:
    - float: The sentiment score (0 to 1, where higher is more positive)
    """
    # Convert input text to sequences of integers
    user_sequences = tokenizer.texts_to_sequences([input])
    # Pad sequences to ensure uniform length
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    # Make prediction using the model
    prediction = model.predict(user_sequences_matrix)
    # Return rounded prediction score
    return round(float(prediction[0][0]), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle both GET and POST requests to the main page.
    For POST requests, perform sentiment analysis and return results.
    For GET requests, just render the form.
    """
    if request.method == "POST":
        # Get user input from form
        text = request.form.get("user_text")
        # Perform VADER sentiment analysis
        vader_sentiment = analyzer.polarity_scores(text)
        # Perform custom model sentiment analysis
        custom_sentiment = sentiment_analysis(text)
        
        # Combine VADER and custom model results
        sentiment = {
            "vader": vader_sentiment,
            "custom_model_positive": custom_sentiment
        }
        
        # Render template with results
        return render_template('form.html', sentiment=sentiment, text=text)
    
    # For GET requests, just render the form
    return render_template('form.html')

# Run the Flask application
# Note: In production, you typically wouldn't run the app directly like this
app.run(debug=True)