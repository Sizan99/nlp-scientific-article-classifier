import pandas as pd
import json
import os
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK data (quietly)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def load_arxiv_data(filepath, max_records=10000):
    print(f"Loading data from {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    df = df[['id', 'title', 'abstract', 'categories']]
    print(f"Loaded {len(df)} records.")
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(clean_tokens)

if __name__ == "__main__":
    print("--- Week 1: Data Exploration and Preprocessing ---")
    
    dataset_path = "arxiv-metadata-oai-snapshot.json"
    
    if os.path.exists(dataset_path):
        # 1. Load Data
        df = load_arxiv_data(dataset_path, max_records=15000)
        
        # 2. Handle Categories (Assign a single main category)
        # The categories field is space-separated. We take the first one as the main category.
        df['main_category'] = df['categories'].apply(lambda x: str(x).split(' ')[0])
        
        # 3. Data Exploration & Missing Values
        print("\nChecking for missing values:")
        print(df.isnull().sum())
        
        # --- Visualisation 1: Category Distribution ---
        plt.figure(figsize=(12, 6))
        category_counts = df['main_category'].value_counts().head(20)
        category_counts.plot(kind='bar', color='skyblue')
        plt.title('Top 20 Paper Categories in Dataset')
        plt.xlabel('Category')
        plt.ylabel('Number of Papers')
        plt.tight_layout()
        plt.savefig('visualisation1_categories.png')
        print("\nSaved Visualisation 1: visualisation1_categories.png")
        
        # --- Visualisation 2: Abstract Length Distribution ---
        df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10, 6))
        plt.hist(df['abstract_word_count'], bins=50, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Abstract Word Counts')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('visualisation2_abstract_length.png')
        print("Saved Visualisation 2: visualisation2_abstract_length.png")
        
        # 4. Text Preprocessing
        print("\nCleaning text data (this will take a moment)...")
        df['clean_abstract'] = df['abstract'].apply(clean_text)
        df['clean_title'] = df['title'].apply(clean_text)
        
        # 5. Label Encoding for Categories
        print("Applying Label Encoding to categories...")
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['main_category'])
        
        # 6. Prepare Final Dataset exactly as requested
        # We rename columns to match the brief: Paper id, Title, Abstract, Category (in numbers)
        final_df = pd.DataFrame({
            'Paper id': df['id'],
            'Title': df['clean_title'],
            'Abstract': df['clean_abstract'],
            'Category (in numbers)': df['category_encoded']
        })
        
        # Save to CSV
        output_file = "dataset.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\nSUCCESS: Cleaned dataset saved to {output_file}")
        print("This file contains exactly the columns requested in the Week 1 guide.")
        
    else:
        print(f"Error: {dataset_path} not found.")
