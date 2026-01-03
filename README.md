# AI-Powered Data Science Mock Interviewer

An intelligent voice-driven mock interview system that generates data science questions from "The Hundred-Page Machine Learning Book", listens to your answers, and evaluates your performance using AI.

<img width="1911" height="903" alt="image" src="https://github.com/user-attachments/assets/fea6f6d8-bddc-40a2-9ed4-9e64093adb45" />
<img width="1919" height="925" alt="image" src="https://github.com/user-attachments/assets/cf35809f-2568-42a1-8bd3-59b00b537700" />


## Features

- 🎤 **Voice Interview Mode**: Full voice interaction - the AI speaks questions, and you record your answers.
- 📚 **Knowledge Base**: Extracts concepts from "The Hundred-Page Machine Learning Book".
- 🤖 **AI-Powered Questions**: Generates contextual questions using **Llama 3 (via Groq)**.
- 🗣️ **Real-Time Audio**: 
  - **Text-to-Speech**: High-quality voice via **EdgeTTS**.
  - **Speech-to-Text**: Accurate transcription via **Groq Whisper**.
- 🎯 **Smart Evaluation**: AI evaluates relevance of every answer and provides a final comprehensive scorecard.
- 🔐 **Accounts**: Secure Signup/Login backed by MongoDB.

## Prerequisites

- Python 3.10+
- A running MongoDB instance (`MONGO_URI`)
- **Groq API Key** (`GROQ_API_KEY`) for LLM and Whisper
- The PDF file: `The Hundred-Page Machine Learning Book.pdf` in the project root

## Quick Start

### 1) Clone the repository
```bash
git clone https://github.com/agnihotri-anxh/AI-Powered-Data-Science-Mock-Interviewer.git
cd AI-Powered-Data-Science-Mock-Interviewer
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
MONGO_URI=your_mongodb_connection_string
SECRET_KEY=your_random_secret_string
# Optional Email Config for OTPs
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### 4) Build the knowledge base
```bash
python run_extraction.py
```

### 5) Run the application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## How it works

1.  **Ingestion**: `Data_Ingestion.py` processes the PDF and creates FAISS embeddings.
2.  **Voice Flow**:
    -   The system greets you with **audio**.
    -   It asks **5 questions** sequentially.
    -   You **record** your answer via microphone.
    -   Your audio is transcribed using **Whisper (Groq)**.
    -   The AI evaluates relevance immediately.
3.  **Evaluation**: After 5 questions, a final feedback report is generated summarizing your strengths and weaknesses.

## Tech Stack

-   **Backend**: Flask (Python)
-   **LLM & Transcription**: Groq (Llama 3 & Whisper)
-   **TTS**: EdgeTTS (Free, high quality)
-   **Vector DB**: FAISS
-   **Database**: MongoDB
-   **Frontend**: HTML/JS + TailwindCSS

## Project Structure

```
AI Powered Data Scince Interviewer/
├── app.py                         # Main application entry point
├── config.py                      # Configuration management
├── services/                      # Modular Business Logic
│   ├── database.py                # MongoDB connection
│   ├── llm.py                     # LangChain & Groq Logic
│   ├── audio.py                   # TTS and Transcription services
│   └── email_service.py           # Email handling
├── Data_Ingestion.py              # PDF Processor
├── run_extraction.py              # KB Build script
├── requirements.txt               # Dependencies
├── Procfile                       # Deployment config (Render)
└── templates/                     # HTML Frontend
```

## License

This project is for educational purposes. Please respect the copyright of "The Hundred-Page Machine Learning Book".
