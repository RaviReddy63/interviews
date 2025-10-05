"""
Complete Text Classification Pipeline
Preprocessing Text Data and Training Multiple Models
"""

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Model training
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_fscore_support)

import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set random seed
np.random.seed(42)

# ============================================================================
# STEP 1: CREATE SAMPLE TEXT DATA
# ============================================================================

print("=" * 80)
print("STEP 1: CREATING SAMPLE TEXT DATA")
print("=" * 80)

# Sample product reviews dataset
data = {
    'text': [
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible quality. Waste of money. Do not buy.",
        "Good value for the price. Works as expected.",
        "Horrible experience. Product broke after one week.",
        "Love it! Exceeded my expectations. Highly recommend.",
        "Not satisfied. Poor customer service and low quality.",
        "Decent product. Nothing special but does the job.",
        "Fantastic! Worth every penny. Will buy again.",
        "Disappointing. Expected much better for this price.",
        "Excellent quality and fast shipping. Very happy!",
        "Complete garbage. Returning it immediately.",
        "Pretty good. Minor issues but overall satisfied.",
        "Outstanding! Best in its category. Five stars.",
        "Mediocre at best. Wouldn't recommend to others.",
        "Incredible product. Changed my life for the better.",
        "Worst purchase ever. Total waste of time and money.",
        "Satisfied with my purchase. Good quality product.",
        "Awful. Cheaply made and doesn't work properly.",
        "Great buy! Exactly what I needed. Thank you!",
        "Regret buying this. Save your money.",
        "Amazing quality and design. Love everything about it!",
        "Not worth it. Better alternatives available elsewhere.",
        "Perfect! No complaints whatsoever. Highly satisfied.",
        "Broken on arrival. Very disappointed with this.",
        "Superb product. Elegant design and great functionality.",
        "Barely works. Customer support was unhelpful too.",
        "Really happy with this purchase. Good investment.",
        "Defective item received. Poor quality control.",
        "Brilliant! Solves all my problems. Couldn't be happier.",
        "Absolute trash. Would give zero stars if possible.",
        "Nice product overall. Some room for improvement.",
        "Nightmare experience. Never ordering from here again.",
        "Wonderful quality. Beautifully packaged and delivered.",
        "Junk. Falls apart easily. Very frustrating.",
        "Impressive! Better than advertised. Great deal.",
        "Unacceptable quality. Demanding a full refund.",
        "Solid purchase. Reliable and durable product.",
        "Pathetic. Doesn't match the description at all.",
        "Delighted with this! Perfect for my needs.",
        "Horrible material. Feels cheap and flimsy."
    ],
    'category': [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'neutral', 'positive', 'negative', 'positive',
        'negative', 'neutral', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative',
        'neutral', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative'
    ]
}

df = pd.DataFrame(data)

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nCategory distribution:")
print(df['category'].value_counts())
print(f"\nCategory percentages:")
print(df['category'].value_counts(normalize=True) * 100)

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Text length analysis
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\nText Statistics:")
print(df[['text_length', 'word_count']].describe())

print("\nAverage text length by category:")
print(df.groupby('category')['text_length'].mean())

print("\nAverage word count by category:")
print(df.groupby('category')['word_count'].mean())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# ============================================================================
# STEP 3: TEXT PREPROCESSING - CLEANING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: TEXT PREPROCESSING - CLEANING")
print("=" * 80)

def clean_text(text):
    """Clean text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

print("\nExample of text cleaning:")
print(f"Original: {df['text'].iloc[0]}")
print(f"Cleaned:  {df['cleaned_text'].iloc[0]}")

# ============================================================================
# STEP 4: TEXT PREPROCESSING - TOKENIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: TEXT PREPROCESSING - TOKENIZATION")
print("=" * 80)

def tokenize_text(text):
    """Tokenize text into words"""
    tokens = word_tokenize(text)
    return tokens

# Apply tokenization
df['tokens'] = df['cleaned_text'].apply(tokenize_text)

print("\nExample of tokenization:")
print(f"Text: {df['cleaned_text'].iloc[0]}")
print(f"Tokens: {df['tokens'].iloc[0]}")

# ============================================================================
# STEP 5: TEXT PREPROCESSING - REMOVE STOPWORDS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: TEXT PREPROCESSING - REMOVE STOPWORDS")
print("=" * 80)

# Get English stopwords
stop_words = set(stopwords.words('english'))

print(f"\nNumber of stopwords: {len(stop_words)}")
print(f"Sample stopwords: {list(stop_words)[:10]}")

def remove_stopwords(tokens):
    """Remove stopwords from tokens"""
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Apply stopword removal
df['tokens_no_stop'] = df['tokens'].apply(remove_stopwords)

print("\nExample of stopword removal:")
print(f"Before: {df['tokens'].iloc[0]}")
print(f"After:  {df['tokens_no_stop'].iloc[0]}")

# ============================================================================
# STEP 6: TEXT PREPROCESSING - LEMMATIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: TEXT PREPROCESSING - LEMMATIZATION")
print("=" * 80)

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    """Lemmatize tokens to their base form"""
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized

# Apply lemmatization
df['tokens_lemmatized'] = df['tokens_no_stop'].apply(lemmatize_tokens)

print("\nExample of lemmatization:")
print(f"Before: {df['tokens_no_stop'].iloc[0]}")
print(f"After:  {df['tokens_lemmatized'].iloc[0]}")

# Alternative: Stemming
stemmer = PorterStemmer()

def stem_tokens(tokens):
    """Stem tokens"""
    stemmed = [stemmer.stem(word) for word in tokens]
    return stemmed

df['tokens_stemmed'] = df['tokens_no_stop'].apply(stem_tokens)

print("\nComparison - Lemmatization vs Stemming:")
print(f"Lemmatized: {df['tokens_lemmatized'].iloc[0]}")
print(f"Stemmed:    {df['tokens_stemmed'].iloc[0]}")

# ============================================================================
# STEP 7: CONVERT BACK TO TEXT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: FINAL PREPROCESSED TEXT")
print("=" * 80)

# Join tokens back to text
df['processed_text'] = df['tokens_lemmatized'].apply(lambda x: ' '.join(x))

print("\nPreprocessing pipeline complete!")
print("\nExample comparison:")
print(f"Original:   {df['text'].iloc[0]}")
print(f"Processed:  {df['processed_text'].iloc[0]}")

# ============================================================================
# STEP 8: FEATURE EXTRACTION - TF-IDF
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: FEATURE EXTRACTION - TF-IDF")
print("=" * 80)

# Prepare data
X = df['processed_text']
y = df['category']

# Method 1: TF-IDF Vectorizer (recommended)
tfidf_vectorizer = TfidfVectorizer(
    max_features=100,  # Top 100 features
    min_df=1,          # Minimum document frequency
    max_df=0.8,        # Maximum document frequency
    ngram_range=(1, 2) # Unigrams and bigrams
)

X_tfidf = tfidf_vectorizer.fit_transform(X)

print(f"\nTF-IDF Feature Matrix Shape: {X_tfidf.shape}")
print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")
print(f"\nTop 20 features:")
print(tfidf_vectorizer.get_feature_names_out()[:20])

# Method 2: Count Vectorizer (Bag of Words)
count_vectorizer = CountVectorizer(
    max_features=100,
    ngram_range=(1, 2)
)

X_count = count_vectorizer.fit_transform(X)

print(f"\nCount Vectorizer Shape: {X_count.shape}")

# We'll use TF-IDF for training
X_features = X_tfidf

# ============================================================================
# STEP 9: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print(f"\nTraining set category distribution:")
print(pd.Series(y_train).value_counts())

# ============================================================================
# STEP 10: TRAIN MULTIPLE MODELS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: TRAINING MULTIPLE MODELS")
print("=" * 80)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy
    }
    
    print(f"✓ {name} trained - Accuracy: {accuracy:.4f}")

# ============================================================================
# STEP 11: MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: MODEL EVALUATION")
print("=" * 80)

# Compare all models
print("\nModel Comparison:")
print("-" * 50)
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()]
}).sort_values('Accuracy', ascending=False)

print(comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['predictions']

print(f"\n{'=' * 50}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'=' * 50}")

# Detailed evaluation for best model
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# ============================================================================
# STEP 12: CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 12: CROSS-VALIDATION")
print("=" * 80)

cv_scores = cross_val_score(best_model, X_features, y, cv=5, scoring='accuracy')

print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 13: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 13: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# For Logistic Regression or Linear models
if best_model_name in ['Logistic Regression', 'SVM']:
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    if best_model_name == 'Logistic Regression':
        # Multi-class: get coefficients for each class
        for idx, category in enumerate(best_model.classes_):
            coefficients = best_model.coef_[idx]
            top_indices = np.argsort(np.abs(coefficients))[-10:][::-1]
            
            print(f"\nTop 10 features for '{category}':")
            for i in top_indices:
                print(f"  {feature_names[i]:.<30} {coefficients[i]:.4f}")

# ============================================================================
# STEP 14: PREDICT NEW TEXT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 14: PREDICTING NEW TEXT")
print("=" * 80)

# New sample texts
new_texts = [
    "This is absolutely wonderful! Best thing I've ever bought!",
    "Terrible product. Complete waste of money.",
    "It's okay, nothing special but acceptable."
]

print("\nPredictions on new texts:")
print("-" * 80)

for text in new_texts:
    # Preprocess
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens_filtered = remove_stopwords(tokens)
    tokens_processed = lemmatize_tokens(tokens_filtered)
    processed = ' '.join(tokens_processed)
    
    # Vectorize
    text_vectorized = tfidf_vectorizer.transform([processed])
    
    # Predict
    prediction = best_model.predict(text_vectorized)[0]
    
    # Get probability if available
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(text_vectorized)[0]
        confidence = max(proba)
        print(f"\nText: {text}")
        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence: {confidence:.2%}")
    else:
        print(f"\nText: {text}")
        print(f"Prediction: {prediction.upper()}")

# ============================================================================
# STEP 15: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 15: CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(15, 10))

# Plot 1: Model Comparison
plt.subplot(2, 3, 1)
plt.barh(comparison_df['Model'], comparison_df['Accuracy'])
plt.xlabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xlim([0, 1])
for i, v in enumerate(comparison_df['Accuracy']):
    plt.text(v, i, f' {v:.3f}', va='center')
plt.grid(True, alpha=0.3)

# Plot 2: Category Distribution
plt.subplot(2, 3, 2)
df['category'].value_counts().plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot 4: Text Length Distribution
plt.subplot(2, 3, 4)
for category in df['category'].unique():
    subset = df[df['category'] == category]['text_length']
    plt.hist(subset, alpha=0.5, label=category, bins=10)
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Text Length Distribution by Category')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Word Count Distribution
plt.subplot(2, 3, 5)
for category in df['category'].unique():
    subset = df[df['category'] == category]['word_count']
    plt.hist(subset, alpha=0.5, label=category, bins=10)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Word Count Distribution by Category')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Top Words (from TF-IDF)
plt.subplot(2, 3, 6)
feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(X_tfidf.toarray()).flatten()[::-1]
top_n = 15
top_features = feature_array[tfidf_sorting][:top_n]
top_scores = X_tfidf.toarray().flatten()[tfidf_sorting][:top_n]

plt.barh(range(top_n), top_scores)
plt.yticks(range(top_n), top_features)
plt.xlabel('TF-IDF Score')
plt.title('Top 15 Features by TF-IDF')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('text_classification_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'text_classification_analysis.png'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TEXT CLASSIFICATION PIPELINE COMPLETE!")
print("=" * 80)
print(f"""
✓ Dataset: {df.shape[0]} samples
✓ Categories: {df['category'].nunique()} ({', '.join(df['category'].unique())})
✓ Features: {X_features.shape[1]} TF-IDF features
✓ Models Trained: {len(models)}
✓ Best Model: {best_model_name}
✓ Best Accuracy: {results[best_model_name]['accuracy']:.4f}
✓ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})

Preprocessing Steps Applied:
  1. Text Cleaning (lowercase, remove punctuation, URLs, etc.)
  2. Tokenization
  3. Stopword Removal
  4. Lemmatization
  5. TF-IDF Vectorization

Models Compared:
  • Naive Bayes
  • Logistic Regression
  • Random Forest
  • Support Vector Machine (SVM)
""")
