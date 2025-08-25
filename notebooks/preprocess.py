def select_relevant_columns(df):
    """
    Keep only text and target columns needed for modeling.
    """
    return df[['text', 'gender']].dropna()

def encode_labels(df):
    """
    Map gender labels to integers.
    Example: {'male': 0, 'female': 1}
    """
    mapping = {'male': 0, 'female': 1}
    df = df[df['gender'].isin(mapping.keys())]  # filter out unknowns
    df['label'] = df['gender'].map(mapping)
    return df

import re

def clean_text(text):
    """
    Clean raw text: lowercase, remove URLs, mentions, hashtags, and non-alphabetic characters.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)     # remove URLs
    text = re.sub(r"@\w+", "", text)        # remove mentions
    text = re.sub(r"#\w+", "", text)        # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)    # keep only letters and spaces
    return text.strip()

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_tokens(text):
    """
    Tokenize, remove stopwords, and lemmatize.
    """
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_sequences(texts, vocab_size=10000, max_len=100):
    """
    Tokenize and pad sequences for deep learning models.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return tokenizer, padded

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(texts, max_features=5000):
    """
    Convert text into TF-IDF vectors for classical ML models.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X
import os

def save_processed(df, filename):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(f"data/processed/{filename}", index=False)
