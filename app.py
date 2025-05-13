from flask import Flask, render_template, request
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from dotenv import load_dotenv

load_dotenv()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Load the trained ensemble sentiment analysis model
sentiment_model = None
try:
    with open('ensemble_sentiment_model.pkl', 'rb') as model_file:
        sentiment_model = pickle.load(model_file)
    print("Ensemble sentiment model loaded successfully.")
except Exception as e:
        print(f"Error loading ensemble sentiment model: {e}")

# Load the TF-IDF vectorizer
tfidf_vectorizer = None
try:
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    print("TF-IDF vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF vectorizer: {e}")

def preprocess_text(text):
    """
    Preprocesses the input text by removing URLs, mentions, hashtags,
    special characters, converting to lowercase, tokenizing, lemmatizing,
    and removing stop words.

    Args:
        text: The text string to preprocess.

    Returns:
        A string containing the preprocessed text.
    """
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def predict_sentiment(text):
    """
    Predicts the sentiment of the given text using the loaded sentiment analysis model.

    Args:
        text: The text string for which to predict the sentiment.

    Returns:
        A string representing the predicted sentiment ('Negative', 'Neutral', 'Positive'),
        or an error message if the model or vectorizer failed to load.
    """
    if sentiment_model is None or tfidf_vectorizer is None:
        return "Error: Sentiment model or vectorizer not loaded.  Please check the application logs."
    processed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([processed_text])
    prediction = sentiment_model.predict(text_tfidf)[0]
    if prediction == 0:
        return 'Negative'
    elif prediction == 1:
        return 'Neutral'
    else:
        return 'Positive'

def analyze_csv_data(csv_file_path, product_name, text_column_name='text'):
    """
    Analyzes sentiment of text data from a CSV file, focusing on a specific product.

    Args:
        csv_file_path (str): Path to the CSV file.
        product_name (str): The name of the product to search for.
        text_column_name (str): The name of the column containing the text to analyze.
            Defaults to 'text'.

    Returns:
        pandas.DataFrame: A DataFrame containing the original data, the processed text,
            and the predicted sentiment, filtered for the product, or None if an error occurs.
    """
    try:
        data = pd.read_csv(csv_file_path, encoding='latin1')
        if text_column_name not in data.columns:
            print(f"Error: Column '{text_column_name}' not found in CSV file.")
            return None

        # Create a new column 'text_lower' for case-insensitive search
        data['text_lower'] = data[text_column_name].str.lower()

        # Filter the DataFrame for rows containing the product name
        product_data = data[data['text_lower'].str.contains(product_name.lower(), na=False)]

        if product_data.empty:
            print(f"No data found for product: {product_name}")
            return pd.DataFrame()  # Return empty DataFrame to avoid errors

        product_data = product_data[[text_column_name]].dropna()
        if product_data.empty:
            print(f"No valid data to analyze in the CSV file for the product: {product_name}")
            return pd.DataFrame()

        product_data['processed_text'] = product_data[text_column_name].apply(preprocess_text)
        product_data['sentiment'] = product_data['processed_text'].apply(predict_sentiment)
        return product_data

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the CSV file: {e}")
        return None



@app.route('/')
def home():
    """
    Renders the home page (index.html).
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request when the user submits a CSV file and product name.
    Analyzes the data from the CSV, predicts sentiment, and renders the results page.
    """
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return render_template('result.html', error_message="No file part")
        file = request.files['csv_file']
        if file.filename == '':
            return render_template('result.html', error_message="No selected file")

        product_name = request.form.get('product_name', '')  # Get product name
        if not product_name:
            return render_template('result.html', error_message="Product name is required")

        if file:
            # Save the file temporarily.
            csv_file_path = "temp.csv"
            file.save(csv_file_path)

            text_column = request.form.get('text_column', 'text')
            analysis_results = analyze_csv_data(csv_file_path, product_name, text_column)

            # Clean up the temporary file
            try:
                import os
                os.remove(csv_file_path)
            except Exception as e:
                print(f"Error deleting temporary file: {e}")

            if analysis_results is None:
                return render_template('result.html', error_message="Error processing CSV file.  Check file and column name.")

            if analysis_results.empty:
                return render_template('result.html', error_message=f"No data found for product: {product_name}")

            original_texts = analysis_results[text_column].tolist()
            predicted_sentiments = analysis_results['sentiment'].tolist()
            output = dict(zip(original_texts, predicted_sentiments))

            Neucount = predicted_sentiments.count('Neutral')
            Negcount = predicted_sentiments.count('Negative')
            Poscount = predicted_sentiments.count('Positive')

            # Basic emotion analysis (from previous code) - Consider refactoring con1
            all_cleaned_text = " ".join(analysis_results['processed_text'].tolist())
            emo = con1(all_cleaned_text)
            h = emo.count(' happy')
            s = emo.count(' sad')
            a = emo.count(' angry')
            l = emo.count(' loved')
            pl = emo.count(' powerless')
            su = emo.count(' surprise')
            fl = emo.count(' fearless')
            c = emo.count(' cheated')
            at = emo.count(' attracted')
            so = emo.count(' singled out')
            ax = emo.count(' anxious')

            # Determine ad suitability
            if Poscount > Negcount and Poscount > Neucount:
                ad_suitability = "It is a good time to post an ad for this product."
            elif Negcount > Poscount and Negcount > Neucount:
                ad_suitability = "It is not a good time to post an ad for this product."
            else:
                ad_suitability = "The sentiment around this product is neutral. Proceed with caution."

            return render_template('result.html', outputs=output, NU=Neucount, N=Negcount,
                                   P=Poscount, happy=h, sad=s, angry=a, loved=l,
                                   powerless=pl, surprise=su, fearless=fl, cheated=c,
                                   attracted=at, singledout=so, anxious=ax,
                                   original_texts=original_texts, product_name=product_name, ad_suitability=ad_suitability)  # Pass product name and ad_suitability

    else:
        return render_template('index.html')

def con1(sentence):
    """
    Performs a simple emotion analysis on the input sentence based on a word list.
    This function reads an 'emotions.txt' file, which should contain words
    and their associated emotions (e.g., "happy:joy").  It then checks if any of
    those words are present in the input sentence and returns the corresponding emotions.

    Args:
        sentence: The input sentence to analyze.

    Returns:
        A list of emotions found in the sentence.  Returns an empty list if
        the 'emotions.txt' file is not found or if no matching words are found.
    """
    emotion_list = []
    sentence = sentence.split(' ')
    try:
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')
                if word in sentence:
                    emotion_list.append(emotion)
    except FileNotFoundError:
        print("Warning: emotions.txt not found. Emotion analysis will be skipped.")
        return []
    return emotion_list

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
