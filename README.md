# AI-Powered Data Science Mock Interviewer

An intelligent mock interview system that generates data science questions from "The Hundred-Page Machine Learning Book" and evaluates your responses using AI.

## Features

- 📚 **Knowledge Base**: Extracts concepts from "The Hundred-Page Machine Learning Book"
- 🤖 **AI-Powered Questions**: Generates contextual questions using extracted knowledge
- 🎯 **Smart Evaluation**: AI evaluates responses with detailed feedback
- 🔐 **Accounts**: Signup/login backed by MongoDB
- 🎵 **Audio Support**: Text-to-speech for questions via ElevenLabs

## Prerequisites

- Python 3.9+ (tested on 3.10/3.11)
- A running MongoDB instance and connection string (`MONGO_URI`)
- API keys: Groq (`GROQ_API_KEY`) and ElevenLabs (`ELEVENLABS_API_KEY`)
- The PDF file: `The Hundred-Page Machine Learning Book.pdf` in the project root

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` file in the project root. On Windows PowerShell:

```powershell
Copy-Item env_example.txt .env
```

Then edit `.env` and set at least:

- `GROQ_API_KEY=<your_groq_api_key>`
- `ELEVENLABS_API_KEY=<your_elevenlabs_api_key>`
- `MONGO_URI=<your_mongodb_connection_string>`
- `SECRET_KEY=<any_random_secret_string>`
- Optional: `MONGODB_DB` (default: `AI-Interviewer-DB`), `MONGODB_USERS_COLLECTION` (default: `users`)

### 3) Build the knowledge base

Make sure the PDF is present as `The Hundred-Page Machine Learning Book.pdf` and run:

```bash
python run_extraction.py
```

This will create `knowledge_base/index.faiss` and `knowledge_base/index.pkl`.

### 4) Start the app

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

## How it works

- `Data_Ingestion.py`: Extracts text from the PDF and builds a FAISS vector store using sentence-transformers.
- `app.py`: Flask app that
  - serves pages (`landing_page.html`, `login.html`, `signup.html`, `index.html`),
  - uses LangChain + Groq to generate questions from the knowledge base,
  - checks answer relevance and composes a final evaluation after 10 Q&A pairs,
  - uses ElevenLabs to synthesize audio for questions at `/synthesize`.

### Routes

- `GET /` → Landing page
- `GET|POST /login`, `GET|POST /signup` → Authentication
- `GET /interview` → Interview UI (requires login)
- `POST /ask` → Generate a question for a given topic
- `POST /submit_answer` → Submit answer, get follow-up or final feedback
- `POST /synthesize` → TTS audio for given text (requires login)

## Project structure

```
AI Powered Data Scince Interviewer/
├── app.py                         # Flask application
├── Data_Ingestion.py              # PDF ingestion and vector store builder
├── run_extraction.py              # Builds the knowledge base
├── requirements.txt               # Python dependencies
├── env_example.txt                # .env template
├── templates/
│   ├── index.html                 # Interview UI
│   ├── landing_page.html          # Landing page
│   ├── login.html                 # Login page
│   └── signup.html                # Signup page
├── knowledge_base/
│   ├── index.faiss                # FAISS index
│   └── index.pkl                  # Embeddings/metadata
└── The Hundred-Page Machine Learning Book.pdf
```

## Usage

1. Sign up at `/signup` and then log in at `/login`.
2. Go to `/interview` and enter a data science topic (e.g., "cross validation").
3. Answer each question. After 10 questions, you’ll receive a detailed final evaluation.
4. Use the audio button (if available) to hear questions synthesized via ElevenLabs.

## Troubleshooting

- **Knowledge base not found**: Run `python run_extraction.py` and ensure the PDF filename matches exactly.
- **MongoDB connection failed**: Verify `MONGO_URI` and that your database allows connections from your machine.
- **Missing API keys**: Ensure `GROQ_API_KEY` and `ELEVENLABS_API_KEY` are present in `.env`.
- **Port in use**: Change the port in `app.py` (default 5000).
- **Windows venv**: If using a venv, activate it with `./venv/Scripts/Activate.ps1` before installing/running.

## Technology stack

- **Backend**: Python, Flask
- **AI Orchestration**: LangChain
- **LLM**: Groq (`llama-3.1-8b-instant`)
- **Embeddings & Vector Search**: sentence-transformers, FAISS
- **Auth & Data**: Flask-Login session + MongoDB (`flask-pymongo`)
- **TTS**: ElevenLabs
- **PDF Processing**: PyPDF / related utilities

## License

This project is for educational purposes. Please respect the copyright of
"The Hundred-Page Machine Learning Book".
