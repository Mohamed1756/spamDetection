import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

class CustomEmailSpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

    def preprocess_text(self, text):
        # Remove email headers
        text = re.sub(r'From.*?Subject:', '', text, flags=re.DOTALL)

        # Convert to lowercase
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Remove special characters and stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

        # Perform stemming using PorterStemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        # Join the tokens back to form the preprocessed text
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def train_classifier(self, csv_file_path):
        # Load the CSV data into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Apply preprocessing to the 'text' column
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # Transform the preprocessed text data into numerical vectors
        X = self.vectorizer.fit_transform(df['processed_text'])

        # Get the target labels
        y = df['target']

        # Split the dataset into training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the classifier on the training data
        self.classifier.fit(X_train, y_train)

    def predict_email(self, email_text):
        # Preprocess the text of the email
        preprocessed_email = self.preprocess_text(email_text)
        X_email = self.vectorizer.transform([preprocessed_email])

        # Predict the label of the email (0 for not spam, 1 for spam)
        predicted_label = self.classifier.predict(X_email)[0]

        return predicted_label


# Example usage:
if __name__ == "__main__":
    detector = CustomEmailSpamDetector()

    # Train the classifier on the given CSV file
    detector.train_classifier('/Users/mohamed/Desktop/spam_assassin.csv')

    # Test with a sample email text
    email_text = ("Congratulations! We would like to offer you a unique opportunity to receive\n"
              "a brand new Oral B iO Series 9! To claim, simply take this short survey about your experience with us.")

    predicted_label = detector.predict_email(email_text)
    print("")
    print("")
    print ("*****EMAIL TEST*****")
    print("")
    print (email_text)
    print("")
    print("----------------------------------------------")
    if predicted_label == 0:
        print("The email is NOT spam.")
    else:
        print("\033[1mRESULT:\033[0m This email is spam.")
