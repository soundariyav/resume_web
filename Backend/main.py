from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import re
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
# Download necessary NLTK data (if not already downloaded)
import nltk
from fastapi.middleware.cors import CORSMiddleware
import os

# Your FastAPI app
app = FastAPI()

# CORS configuration


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join tokens back into a cleaned string
    return ' '.join(tokens)
# Load BERT model & tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# FastAPI app
app = FastAPI()

# Mean pooling function for BERT embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to get BERT embeddings
def get_bert_embeddings(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).numpy()

# Compute cosine similarity
def compute_similarity(text1, text2):
    embedding1 = get_bert_embeddings(text1)
    embedding2 = get_bert_embeddings(text2)
    score = cosine_similarity(embedding1, embedding2)[0][0] * 100
    return round(score)

# OpenAI API Key (Replace with your API key)
load_dotenv()

# Get the API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Use the API key in your code
openai.api_key = openai_api_key
#openai.api_key =  "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"


# Generate resume using OpenAI LLM
def optimize_resume(existing_resume,job_description):
    prompt = f"""Improve the given resume to be a strong match for the provided job description. 
    - Highlight relevant skills, experiences, and achievements.
    - Remove unrelated information.
    - Use professional, concise, and impactful language.
    - Ensure the resume remains formatted appropriately.

    Existing Resume:
    {existing_resume}

    Job Description:
    {job_description}

    Provide the improved resume.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert career coach and resume writer."},
                  {"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

def preprocess_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([char if char.isalnum() else ' ' for char in text])
    words = [word for word in text.split() if word not in stopwords.words('english')]
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in words]
    return " ".join(stemmed)

# API Endpoint to get relevance score
@app.post("/relevance_score")
async def relevance_score(resume_text: str = Form(...), job_description: str = Form(...)):
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)
    score = compute_similarity(cleaned_resume, cleaned_jd)
    return {"relevance_score": score}

# API Endpoint to generate resume using LLM
@app.post("/optimize_resume")
async def optimize_existing_resume(existing_resume: str = Form(...), job_description: str = Form(...)):
    optimized_resume = optimize_resume(existing_resume, job_description)
    return {"optimized_resume": optimized_resume}
# Run the app: `uvicorn main:app --reload`

vectorizer = joblib.load("vectorizer.pkl")
mo = joblib.load("job_category_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
class ResumeInput(BaseModel):
    resume: str
@app.post("/predict-category")
def predict_category(resume: str = Form(...)):
    # Preprocess the resume text to match the format of your training data
    processed_resume = preprocess_text(resume)
    
    # Transform the processed text using the trained vectorizer
    vectorized = vectorizer.transform([processed_resume]).toarray()
    
    # Get a prediction from the model
    prediction = mo.predict(vectorized)
    
    # Convert the numeric prediction back to the actual category (e.g., 'Data Science')
    category = label_encoder.inverse_transform(prediction)[0]
    
    return {"predicted_category": category}

