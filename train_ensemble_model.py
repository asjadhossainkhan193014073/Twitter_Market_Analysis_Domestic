import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run this once)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('running')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocesses the input text by removing URLs, mentions, hashtags,
    special characters, converting to lowercase, tokenizing, lemmatizing,
    and removing stop words.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def train_and_evaluate_model(csv_file_path, text_column_name='text', sentiment_column_name='label', test_size=0.2, random_state=42):
    """
    Trains and evaluates an ensemble sentiment analysis model using data from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.
        text_column_name (str): The name of the column containing the text data.
            Defaults to 'text'.
        sentiment_column_name (str): The name of the column containing the sentiment labels.
            Defaults to 'label'.
        test_size (float): The proportion of the data used for testing. Defaults to 0.2.
        random_state (int):  Controls the shuffling applied to the data before splitting.
            Defaults to 42.

    Returns:
        tuple: (model, vectorizer, accuracy, classification_report)
            - model: The trained VotingClassifier model.
            - vectorizer: The trained TfidfVectorizer.
            - accuracy: The accuracy of the model on the test set.
            - classification_report: The classification report on the test set (string).
            Returns (None, None, None, None) if an error occurs.
    """
    try:
        data = pd.read_csv(csv_file_path, encoding='latin1')
        # Check for the existence of the specified columns.
        if text_column_name not in data.columns:
            print(f"Error: Text column '{text_column_name}' not found in CSV file.")
            return None, None, None, None
        if sentiment_column_name not in data.columns:
            print(f"Error: Sentiment column '{sentiment_column_name}' not found in CSV file.")
            return None, None, None, None

        data = data[[text_column_name, sentiment_column_name]].dropna()  # Select and drop NaNs
        if data.empty:
            print("Error: No valid data remaining after selecting columns and dropping NaNs.")
            return None, None, None, None

        # Map sentiment values if needed and check for invalid values.
        valid_sentiments = [0, 1]
        invalid_sentiments = data[~data[sentiment_column_name].isin(valid_sentiments)]
        if not invalid_sentiments.empty:
            print("Warning: Invalid sentiment values found in CSV. These rows will be dropped:")
            print(invalid_sentiments)
            data = data[data[sentiment_column_name].isin(valid_sentiments)]
            if data.empty:
                print("Error: No valid data remaining after filtering invalid sentiments.")
                return None, None, None, None

        data['processed_text'] = data[text_column_name].apply(preprocess_text)

        X = data['processed_text']
        y = data[sentiment_column_name].astype(int) # Ensure y is int

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Feature Extraction using TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Initialize individual classifiers
        nb_clf = MultinomialNB()
        lr_clf = LogisticRegression(solver='liblinear', random_state=random_state)
        svm_clf = LinearSVC(random_state=random_state)

        # Train individual classifiers
        nb_clf.fit(X_train_tfidf, y_train)
        lr_clf.fit(X_train_tfidf, y_train)
        svm_clf.fit(X_train_tfidf, y_train)

        # Create a VotingClassifier
        voting_clf = VotingClassifier(estimators=[('nb', nb_clf), ('lr', lr_clf), ('svm', svm_clf)], voting='hard')
        voting_clf.fit(X_train_tfidf, y_train)

        # Evaluate the ensemble model
        y_pred = voting_clf.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        return voting_clf, tfidf_vectorizer, accuracy, report

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None, None, None, None
    except KeyError as e:
        print(f"Error: Required column not found in CSV. Check column names. Error: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None



if __name__ == '__main__':
    csv_file_path = 'train-00000-of-00001.csv'  # Changed to the training file
    text_column_name = 'text'  # Changed to the text column name
    sentiment_column_name = 'label'  # Changed to the label column name

    model, vectorizer, accuracy, report = train_and_evaluate_model(csv_file_path, text_column_name, sentiment_column_name)

    if model and vectorizer:  # Only save if training was successful
        # Save the trained ensemble model and the vectorizer
        try:
            with open('ensemble_sentiment_model.pkl', 'wb') as model_file:
                pickle.dump(model, model_file)
            with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
                pickle.dump(vectorizer, vectorizer_file)
            print("Trained ensemble sentiment model and TF-IDF vectorizer saved.")
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
        except Exception as e:
            print(f"Error saving model or vectorizer: {e}")
    else:
        print("Failed to train and evaluate the model.")
