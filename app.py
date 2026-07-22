from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import time
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = FastAPI()

# ---------------------------------------------------------
# Load Models on Startup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Models onto {device}...")

print("--- Training Week 2 Classical ML Models ---")
data = []
with open("arxiv-metadata-oai-snapshot.json", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000: break
        data.append(json.loads(line))
        
df = pd.DataFrame(data)[['abstract', 'categories']].dropna()
df['main_category'] = df['categories'].apply(lambda x: str(x).split(' ')[0].split('.')[0])

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['abstract'].astype(str))
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, df['main_category'])

print("--- Loading Week 3 LLM Models ---")
# 1. Classification Model (Qwen2.5)
qwen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, torch_dtype=torch.float16, device_map="auto")

# 2. Summarisation Model (DistilBART)
bart_model_name = "sshleifer/distilbart-cnn-12-6"
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name, use_safetensors=True).to(device)

print("Models successfully loaded! FastAPI is ready.")

# Serve the static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    abstract: str
    pipeline_mode: str = "LLM"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/api/analyze")
async def analyze_abstract(req: ChatRequest):
    abstract = req.abstract
    pipeline_mode = req.pipeline_mode
    
    if pipeline_mode == "Classical":
        # 1. Classical Classification
        try:
            vec = vectorizer.transform([abstract])
            category_prediction = lr_model.predict(vec)[0]
        except Exception as e:
            category_prediction = "Error predicting category"
            
        # 2. Classical Extractive Summarisation
        try:
            sentences = sent_tokenize(str(abstract))
            if len(sentences) <= 1:
                generated_title = abstract
            else:
                scores = [len(set(sentence.split())) for sentence in sentences]
                top_idx = np.argmax(scores)
                generated_title = sentences[top_idx]
        except Exception as e:
            generated_title = "Error extracting summary"
            
        return {
            "category": category_prediction,
            "title": generated_title
        }

    # ---------------------------------------------------------
    # Generate Classification (Qwen2.5)
    # ---------------------------------------------------------
    prompt = f"What is the most likely scientific category for this abstract? Answer with only the category name.\\n\\nAbstract: {abstract}"
    messages = [
        {"role": "system", "content": "You are a helpful scientific classifier."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_tokenizer(text, return_tensors="pt").to(device)
        outputs = qwen_model.generate(**inputs, max_new_tokens=15, temperature=0.1, do_sample=False)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        category_prediction = qwen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    except Exception as e:
        category_prediction = "Error predicting category"
        print(e)
        
    # ---------------------------------------------------------
    # Generate Summarised Title (DistilBART)
    # ---------------------------------------------------------
    try:
        inputs = bart_tokenizer(abstract, max_length=1024, return_tensors="pt", truncation=True).to(device)
        # Generate a short title (max 60 tokens)
        summary_ids = bart_model.generate(inputs["input_ids"], max_length=60, min_length=10, num_beams=4, early_stopping=True)
        generated_title = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Ensure it doesn't cut off mid-sentence by trimming to the last period (if one exists)
        if "." in generated_title:
            generated_title = generated_title[:generated_title.rfind(".")+1]
    except Exception as e:
        generated_title = "Error generating title"
        print(e)

    # Return the Chatbot's dual response
    return {
        "category": category_prediction,
        "title": generated_title
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8082)
