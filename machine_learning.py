import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# For Summarisation
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def perform_extractive_summarisation(text, top_n=1):
    """Simple extractive summarisation using sentence length and word density."""
    sentences = sent_tokenize(str(text))
    if not sentences:
        return text
    if len(sentences) <= top_n:
        return text
    
    # Very basic scoring: longer sentences with more unique words get higher scores.
    # In a real TF-IDF summariser, we'd build a matrix for the abstract's sentences.
    scores = []
    for sentence in sentences:
        words = sentence.split()
        score = len(set(words)) # Unique word count as a proxy for information density
        scores.append(score)
        
    top_sentence_indices = np.argsort(scores)[-top_n:]
    top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
    return " ".join(top_sentences)

if __name__ == "__main__":
    print("==================================================")
    print("   Week 2: Machine Learning & Summarisation")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # STEP 1 & 2: Feature Extraction & Text Classification
    # ---------------------------------------------------------
    print("Loading cleaned dataset (dataset.csv)...")
    try:
        df = pd.read_csv("dataset.csv")
        # Drop rows with missing values
        df = df.dropna(subset=['Abstract', 'Category (in numbers)'])
    except Exception as e:
        print("Error loading dataset.csv. Make sure Week 1 script ran successfully.")
        exit()

    print("\n--- Feature Extraction (TF-IDF) ---")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['Abstract'].astype(str))
    y = df['Category (in numbers)']

    print("Splitting data into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Training Model 1: Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_preds))

    print("\n--- Training Model 2: Multinomial Naïve Bayes ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, nb_preds):.4f}")
    print("Naïve Bayes Classification Report:")
    print(classification_report(y_test, nb_preds))

    # Save models for Week 4
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("\nSaved Logistic Regression model and Vectorizer for future use.")

    # ---------------------------------------------------------
    # STEP 3 & 4: Summarisation (Extractive vs Abstractive)
    # ---------------------------------------------------------
    print("\n==================================================")
    print("   Summarisation: Extractive vs Abstractive (BART)")
    print("==================================================\n")
    
    # We grab a few original (uncleaned) abstracts from the json to show proper summarisation
    # Pre-trained models (BART) and sentence tokenizers need grammar and punctuation!
    print("Loading 3 examples from the original JSON for summarisation evaluation...")
    sample_data = []
    with open("arxiv-metadata-oai-snapshot.json", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3: break
            sample_data.append(json.loads(line))
            
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("Loading BART model (this may take a minute the first time)...")
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The new version of Transformers blocks loading old .bin files in PyTorch 2.5 for security reasons.
    # We bypass this by forcing the secure .safetensors format!
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)

    for i, paper in enumerate(sample_data):
        print(f"\n--- Example {i+1} ---")
        print(f"ORIGINAL TITLE: {paper['title']}")
        print(f"ORIGINAL ABSTRACT: {paper['abstract'][:300]}...")
        
        # Extractive
        ext_summary = perform_extractive_summarisation(paper['abstract'])
        print(f"\n> EXTRACTIVE SUMMARY (Baseline):")
        print(f"{ext_summary}")
        
        # Abstractive (BART)
        input_length = len(paper['abstract'].split())
        max_len = min(40, max(15, input_length // 2))
        
        try:
            inputs = tokenizer(paper['abstract'], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=max_len, min_length=10, num_beams=4, early_stopping=True)
            bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(f"\n> ABSTRACTIVE SUMMARY (BART - Advanced):")
            print(f"{bart_summary}")
        except Exception as e:
            print(f"\n> ABSTRACTIVE SUMMARY ERROR: {e}")
            
    print("\n==================================================")
    print("Week 2 Pipeline Complete!")
