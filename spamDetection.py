# Importing data manipulation and analysis libraries
# Importing data manipulation and analysis libraries
import pandas as pd
import numpy as np

# Importing machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download the required resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV data into a DataFrame
df = pd.read_csv('/Users/mohamed/Desktop/spam_assassin.csv')

# Explore the dataset
print(df.head())  # Display the first few rows of the DataFrame
print(df.info())  # Get information about the DataFrame (e.g., data types, missing values)

# Check the distribution of spam and non-spam emails
print(df['target'].value_counts())

# Function to preprocess text
def preprocess_text(text):
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

# Apply preprocessing to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)

# Continue from your existing code...
# Apply preprocessing to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)

# Save the preprocessed DataFrame to a new CSV file
df.to_csv('/Users/mohamed/Desktop/spamDetection/preprocessed_spam_assassin.csv', index=False)
