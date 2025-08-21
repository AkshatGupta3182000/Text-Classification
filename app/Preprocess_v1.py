## Importing all the necessary Libraries

import pandas as pd
import warnings
import nltk

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

## downloading stopwords and wordnet from nltk
nltk.download('stopwords')
nltk.download('wordnet')
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Function to preprocess the text data.
    It performs the following steps:
    1. Lowercasing
    2. Removing punctuation
    3. Removing stopwords
    4. Lemmatization
    """
    # convert the input to string
    text = str(text)

    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Tokenize and remove stopwords
    
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)