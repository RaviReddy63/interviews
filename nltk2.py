"""
NLTK Text Classification Pipeline - Complete Guide
Use Case: Sentiment Analysis on Movie Reviews
"""

# ============================================================================
# STEP 1: SETUP AND IMPORTS
# ============================================================================

import nltk
import pandas as pd
import numpy as np
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk import FreqDist
import string
import random

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================

def load_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    # Load movie reviews dataset
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    
    # Shuffle for randomness
    random.shuffle(documents)
    
    print(f"Total documents: {len(documents)}")
    print(f"Categories: {movie_reviews.categories()}")
    
    return documents

# ============================================================================
# STEP 3: DATA CLEANING
# ============================================================================

def clean_text(tokens):
    """Clean text data"""
    
    # Convert to lowercase
    tokens = [w.lower() for w in tokens]
    
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove short words (length < 3)
    tokens = [word for word in tokens if len(word) >= 3]
    
    return tokens

# ============================================================================
# STEP 4: TEXT PREPROCESSING
# ============================================================================

def preprocess_text(tokens):
    """Apply various preprocessing techniques"""
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatization (choose either lemmatization OR stemming)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    # Alternative: Stemming (uncomment to use instead of lemmatization)
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(w) for w in tokens]
    
    return tokens

# ============================================================================
# STEP 5: FEATURE EXTRACTION
# ============================================================================

def extract_features(document, word_features):
    """Extract features for classification"""
    document_words = set(document)
    features = {}
    
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    
    return features

def get_top_words(all_words, n=2000):
    """Get top N most frequent words as features"""
    freq_dist = FreqDist(all_words)
    return list(freq_dist.keys())[:n]

# ============================================================================
# STEP 6: PREPARE TRAINING DATA
# ============================================================================

def prepare_data_for_training(documents):
    """Complete preprocessing pipeline"""
    print("\nPreprocessing data...")
    
    # Clean and preprocess all documents
    processed_docs = []
    all_words = []
    
    for tokens, category in documents:
        # Clean
        cleaned = clean_text(tokens)
        # Preprocess
        processed = preprocess_text(cleaned)
        
        processed_docs.append((processed, category))
        all_words.extend(processed)
    
    # Get word features
    word_features = get_top_words(all_words, n=2000)
    
    # Create feature sets
    featuresets = [(extract_features(doc, word_features), cat) 
                   for (doc, cat) in processed_docs]
    
    return featuresets, word_features

# ============================================================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================================================

def split_data(featuresets, train_ratio=0.8):
    """Split data into training and testing sets"""
    train_size = int(len(featuresets) * train_ratio)
    
    train_set = featuresets[:train_size]
    test_set = featuresets[train_size:]
    
    print(f"\nTraining set size: {len(train_set)}")
    print(f"Testing set size: {len(test_set)}")
    
    return train_set, test_set

# ============================================================================
# STEP 8: TRAIN MODELS
# ============================================================================

def train_models(train_set):
    """Train different classification models"""
    print("\nTraining models...")
    
    # Naive Bayes Classifier
    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    # Decision Tree Classifier
    dt_classifier = nltk.DecisionTreeClassifier.train(train_set)
    
    # Maximum Entropy Classifier (may take longer)
    # maxent_classifier = nltk.MaxentClassifier.train(train_set, max_iter=3)
    
    return nb_classifier, dt_classifier

# ============================================================================
# STEP 9: EVALUATE MODELS
# ============================================================================

def evaluate_models(classifiers, test_set):
    """Evaluate trained models"""
    print("\nModel Performance:")
    print("=" * 50)
    
    names = ['Naive Bayes', 'Decision Tree']
    
    for name, classifier in zip(names, classifiers):
        accuracy = nltk.classify.accuracy(classifier, test_set) * 100
        print(f"{name} Accuracy: {accuracy:.2f}%")
    
    # Show most informative features for Naive Bayes
    print("\nMost Informative Features (Naive Bayes):")
    print("=" * 50)
    classifiers[0].show_most_informative_features(15)

# ============================================================================
# STEP 10: MAKE PREDICTIONS
# ============================================================================

def predict_sentiment(text, classifier, word_features):
    """Predict sentiment of new text"""
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Clean
    cleaned = clean_text(tokens)
    
    # Preprocess
    processed = preprocess_text(cleaned)
    
    # Extract features
    features = extract_features(processed, word_features)
    
    # Predict
    sentiment = classifier.classify(features)
    prob_dist = classifier.prob_classify(features)
    
    return sentiment, prob_dist.prob(sentiment)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Load data
    documents = load_data()
    
    # Step 2-6: Prepare data
    featuresets, word_features = prepare_data_for_training(documents)
    
    # Step 7: Split data
    train_set, test_set = split_data(featuresets)
    
    # Step 8: Train models
    nb_classifier, dt_classifier = train_models(train_set)
    
    # Step 9: Evaluate
    evaluate_models([nb_classifier, dt_classifier], test_set)
    
    # Step 10: Test predictions
    print("\n" + "=" * 50)
    print("TESTING PREDICTIONS")
    print("=" * 50)
    
    test_reviews = [
        "This movie was absolutely fantastic! Great acting and plot.",
        "Terrible film. Waste of time and money.",
        "An okay movie, nothing special but not bad either."
    ]
    
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review, nb_classifier, word_features)
        print(f"\nReview: {review}")
        print(f"Predicted Sentiment: {sentiment.upper()}")
        print(f"Confidence: {confidence:.2%}")

    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print("=" * 50)
