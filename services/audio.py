import os
import asyncio
import edge_tts
from flask import Response, jsonify
from groq import Groq
from config import Config

def transcribe_audio_file(audio_path):
    """Transcribes audio using Groq Whisper"""
    if not Config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing")
    
    client = Groq(api_key=Config.GROQ_API_KEY)
    
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            response_format="json",
            language="en",
            temperature=0.0
        )
    return transcription.text

async def generate_edge_tts_audio(text):
    """Generates audio bytes using Edge TTS"""
    voice = "en-US-AriaNeural"
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def synthesize_audio_response(text):
    """Synchronous wrapper for TTS generation"""
    if len(text) > 5000:
        text = text[:5000] + "..."

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio_data = loop.run_until_complete(generate_edge_tts_audio(text))
        finally:
            loop.close()
            
        if not audio_data:
            raise ValueError("Empty audio stream")

        response = Response(audio_data, mimetype="audio/mpeg")
        response.headers['Content-Length'] = str(len(audio_data))
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": str(e)}), 500
