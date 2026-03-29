"""
src/main.py — FastAPI Backend for Sign Language Web Application
"""
import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import sys
from pathlib import Path
import google.generativeai as genai

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.autocomplete import Autocomplete

# Configure Gemini
genai.configure(api_key=config.GEMINI_API_KEY)
# Use gemini-1.5-flash for speed and efficiency
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title="Sign Language Interpreter API")

# Load Model
MODEL_PATH = config.LANDMARK_MODEL_PATH
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Load Label Map
with open(config.LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx_to_class = {int(v): k for k, v in label_map.items()}

# Autocomplete
ac = Autocomplete()

class LandmarkData(BaseModel):
    landmarks: List[float] # 42 flattened points (x,y for 21 marks)

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class SuggestionRequest(BaseModel):
    text: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: LandmarkData):
    if len(data.landmarks) != 42:
        raise HTTPException(status_code=400, detail="Invalid landmark data length. Expected 42.")
    
    input_batch = np.array([data.landmarks])
    preds = model.predict(input_batch, verbose=0)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx])
    label = idx_to_class.get(idx, "?")
    
    return PredictionResponse(label=label, confidence=confidence)

@app.post("/suggestions")
async def suggestions(req: SuggestionRequest):
    words = req.text.split(" ")
    current_word = words[-1] if words else ""
    suggs = ac.get_suggestions(current_word)
    return {"suggestions": suggs}

class RefineRequest(BaseModel):
    text: str

@app.post("/refine")
async def refine(req: RefineRequest):
    if not req.text.strip():
        return {"refined_text": ""}
    
    prompt = f"""
    Convert the following sequence of sign language characters into a natural, grammatically correct English sentence. 
    Fix common spelling errors, remove duplicate letters that don't belong, and infer intended words. 
    Output ONLY the refined sentence without any preamble or explanation.
    
    Sequence: {req.text}
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        refined_text = response.text.strip()
        # Ensure it's not returning an error or empty
        if not refined_text:
            return {"refined_text": req.text}
        return {"refined_text": refined_text}
    except Exception as e:
        print(f"Gemini Error: {e}")
        # Fallback to original text if Gemini fails
        return {"refined_text": req.text}

# Serve static files
# Ensure public directory exists
PUBLIC_DIR = os.path.join(config.BASE_DIR, "public")
if not os.path.exists(PUBLIC_DIR):
    os.makedirs(PUBLIC_DIR)
    os.makedirs(os.path.join(PUBLIC_DIR, "css"))
    os.makedirs(os.path.join(PUBLIC_DIR, "js"))

app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
