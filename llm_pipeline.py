import pandas as pd
import json
import os
import time
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Qwen2.5-1.5B-Instruct on {device}...")
# We use Qwen2.5-1.5B because it's incredibly smart but small enough to squeeze into 4GB VRAM!
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load in float16 with device_map="auto" to safely stream it into the 4GB GPU without crashing System RAM
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def get_gemini_response(prompt, max_retries=1):
    """Replaced Gemini API with Qwen2.5 Local LLM"""
    # Wrap prompt in Qwen's chat format
    messages = [
        {"role": "system", "content": "You are a helpful scientific assistant. Answer precisely and concisely."},
        {"role": "user", "content": prompt}
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        # Generate with strict length limits so it doesn't crash the GPU
        outputs = llm_model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=False)
        # Extract only the generated answer (not the prompt)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "UNKNOWN"

if __name__ == "__main__":
    print("==================================================")
    print("   Week 3: LLM Pipeline & RAG (using Gemini)")
    print("==================================================\n")

    print("Loading dataset from JSON to get textual categories...")
    data = []
    with open("arxiv-metadata-oai-snapshot.json", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 550: break
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    df = df.rename(columns={'abstract': 'Abstract', 'categories': 'main_category', 'title': 'Title'})
    df['main_category'] = df['main_category'].apply(lambda x: x.split()[0] if isinstance(x, str) else x) 
    df = df.dropna(subset=['Abstract', 'main_category'])
    
    # We will use 500 papers as our "Knowledge Base" for RAG
    # And ONLY 10 papers for our "Test Set" to prevent blowing the Gemini free tier API limits!
    train_df = df.iloc[:500].reset_index(drop=True)
    test_df = df.iloc[500:510].reset_index(drop=True)
    
    # Get all unique categories for the prompt
    categories_list = ", ".join(train_df['main_category'].unique())
    print(f"Loaded {len(train_df)} training (knowledge base) samples and {len(test_df)} test samples.")
    print(f"Unique categories available: {len(train_df['main_category'].unique())}")
    
    # ---------------------------------------------------------
    # STEP 1: LLM Zero-Shot Classification
    # ---------------------------------------------------------
    print("\n--- Running LLM Zero-Shot Classification ---")
    print("This will take a few minutes as we query the Gemini API 50 times...")
    
    zero_shot_preds = []
    true_labels = []
    
    for idx, row in test_df.iterrows():
        prompt = f"""You are a scientific paper classification system. 
Classify the following abstract into exactly ONE of these categories: {categories_list}.
Only output the exact category name, nothing else.

Abstract: {row['Abstract']}
Category:"""
        
        prediction = get_gemini_response(prompt)
        # Clean up prediction just in case LLM added extra text
        prediction = prediction.split()[0] if prediction else "UNKNOWN"
        zero_shot_preds.append(prediction)
        true_labels.append(row['main_category'])
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/50 papers...")
            
    zero_shot_acc = accuracy_score(true_labels, zero_shot_preds)
    print(f"\n> Zero-Shot LLM Classification Accuracy: {zero_shot_acc:.4f} ({(zero_shot_acc*100):.1f}%)")

    # ---------------------------------------------------------
    # STEP 2: LLM Summarisation (Title Generation)
    # ---------------------------------------------------------
    print("\n--- Running LLM Summarisation (Title Generation) ---")
    for i in range(3):
        abstract = test_df.iloc[i]['Abstract']
        true_title = test_df.iloc[i]['Title']
        
        prompt = f"Generate a highly accurate, professional scientific title for the following abstract. Output only the title.\n\nAbstract: {abstract}"
        generated_title = get_gemini_response(prompt)
        
        print(f"\nPaper {i+1}:")
        print(f"ORIGINAL TITLE : {true_title}")
        print(f"LLM GENERATED  : {generated_title}")

    # ---------------------------------------------------------
    # STEP 3: Retrieval-Augmented Generation (RAG)
    # ---------------------------------------------------------
    print("\n--- Running RAG-Enhanced Classification ---")
    print("Building Vector Database from the 500 Knowledge Base papers...")
    
    vectorizer = TfidfVectorizer(max_features=5000)
    kb_vectors = vectorizer.fit_transform(train_df['Abstract'].astype(str))
    
    rag_preds = []
    
    for idx, row in test_df.iterrows():
        # 1. Retrieve most similar documents
        query_vec = vectorizer.transform([str(row['Abstract'])])
        similarities = cosine_similarity(query_vec, kb_vectors).flatten()
        top_indices = similarities.argsort()[-3:][::-1] # Get top 3
        
        # 2. Build RAG Prompt (Few-Shot)
        context = ""
        for i, kb_idx in enumerate(top_indices):
            context += f"Example {i+1}:\nAbstract: {train_df.iloc[kb_idx]['Abstract']}\nCategory: {train_df.iloc[kb_idx]['main_category']}\n\n"
            
        prompt = f"""You are a scientific paper classification system. 
Here are some examples of similar papers and their correct categories:

{context}
Based on these examples, classify the following abstract into exactly ONE category.
Only output the exact category name, nothing else.

Abstract: {row['Abstract']}
Category:"""
        
        prediction = get_gemini_response(prompt)
        prediction = prediction.split()[0] if prediction else "UNKNOWN"
        rag_preds.append(prediction)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/50 papers...")
            
    rag_acc = accuracy_score(true_labels, rag_preds)
    print(f"\n> RAG-Enhanced LLM Classification Accuracy: {rag_acc:.4f} ({(rag_acc*100):.1f}%)")
    
    print("\n==================================================")
    print("Week 3 Pipeline Complete!")
    print(f"Baseline ML (Week 2): 44.0%")
    print(f"Zero-Shot LLM       : {(zero_shot_acc*100):.1f}%")
    print(f"RAG-Enhanced LLM    : {(rag_acc*100):.1f}%")
