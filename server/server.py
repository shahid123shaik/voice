import asyncio
import tempfile
import numpy as np
import pyttsx3
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# -------------------------------
# Config
# -------------------------------
INPUT_LANGUAGES = ["ar", "ml", "ta", "mn"]  # Arabic, Malayalam, Tamil, Mongolian
OUTPUT_LANGUAGES = {"en": "English", "hi": "Hindi", "te": "Telugu"}

CHUNK_SECONDS = 2.0
SAMPLERATE = 16000

# -------------------------------
# Load models
# -------------------------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("Loading translation model...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
trans_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "input_langs": INPUT_LANGUAGES,
                                                     "output_langs": OUTPUT_LANGUAGES})

# -------------------------------
# Helper functions
# -------------------------------
def translate_text(text, tgt_lang):
    tokenizer.src_lang = "auto"
    encoded = tokenizer(text, return_tensors="pt")
    forced_id = tokenizer.get_lang_id(tgt_lang)
    gen = trans_model.generate(**encoded, forced_bos_token_id=forced_id, max_length=512)
    translated = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translated

def tts_to_bytes(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        tts_engine.save_to_file(text, f.name)
        tts_engine.runAndWait()
        data, sr = sf.read(f.name)
    return data, sr

# -------------------------------
# WebSocket for real-time translation
# -------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            # Receive audio chunk as float32 bytes
            msg = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(msg, dtype=np.float32)

            # Save chunk to temporary WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                sf.write(tmp_wav.name, audio_chunk, SAMPLERATE)
                # ASR
                result = whisper_model.transcribe(tmp_wav.name, language="auto")
                text = result.get("text", "").strip()
                if not text:
                    continue

            # Translate to target (here hardcoded "en" for demo)
            translated = translate_text(text, "en")

            # TTS
            data, sr = tts_to_bytes(translated)

            # Send audio back as bytes
            await websocket.send_bytes(data.astype(np.float32).tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")
