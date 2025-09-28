# AI-Powered Data Science Mock Interviewer

An intelligent mock interview system that generates data science questions from "The Hundred-Page Machine Learning Book" and evaluates your responses using AI.
<img width="1829" height="894" alt="Screenshot 2025-09-22 161455" src="https://github.com/user-attachments/assets/413a68cd-5d9f-4bc9-98c1-292d2e008a36" />
<img width="1810" height="884" alt="Screenshot 2025-09-22 162159" src="https://github.com/user-attachments/assets/1f1d646a-d17d-4f8c-bbfe-ba5ffd2f6535" />
<img width="1522" height="865" alt="Screenshot 2025-09-22 162731" src="https://github.com/user-attachments/assets/0e65c8f6-771e-4079-93c1-d95780557a03" />

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

### Option 1: Docker Deployment (Recommended)

#### Prerequisites for Docker
- Docker and Docker Compose installed
- API keys: Groq (`GROQ_API_KEY`) and ElevenLabs (`ELEVENLABS_API_KEY`)
- The PDF file: `The Hundred-Page Machine Learning Book.pdf` in the project root

#### 1) Clone the repository
```bash
git clone https://github.com/agnihotri-anxh/AI-Powered-Data-Science-Mock-Interviewer.git
cd AI-Powered-Data-Science-Mock-Interviewer
```

#### 2) Set up environment variables
Create a `.env` file in the project root:
```bash
# Copy the example file
cp env_example.txt .env
```

Edit `.env` and set your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
SECRET_KEY=your_super_secret_key_here
```

#### 3) Build and run with Docker Compose
```bash
# Build and start all services (Flask app + MongoDB)
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The application will be available at `http://localhost:5000`

#### 4) Build the knowledge base (first time only)
```bash
# Run the knowledge base extraction inside the container
docker-compose exec app python run_extraction.py
```

#### 5) Stop the services
```bash
docker-compose down
```

### Option 2: Local Development

#### 1) Install dependencies

```bash
pip install -r requirements.txt
```

#### 2) Configure environment variables

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

#### 3) Build the knowledge base

Make sure the PDF is present as `The Hundred-Page Machine Learning Book.pdf` and run:

```bash
python run_extraction.py
```

This will create `knowledge_base/index.faiss` and `knowledge_base/index.pkl`.

#### 4) Start the app

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
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── .dockerignore                  # Docker ignore file
├── .gitignore                     # Git ignore file
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
- **LLM**: Groq (`mixtral-8x7b-32768`)
- **Embeddings & Vector Search**: sentence-transformers, FAISS
- **Auth & Data**: Flask-Login session + MongoDB (`flask-pymongo`)
- **TTS**: ElevenLabs
- **PDF Processing**: PyPDF / related utilities

## License

This project is for educational purposes. Please respect the copyright of
"The Hundred-Page Machine Learning Book".
Use it inside an itemize list. Example second bullet:
```latex
\begin{itemize}
  \item Built mock interviewer leveraging RAG over ML book content
  \item Implemented FAISS-based retrieval with Groq LLM; added relevance checks and final scoring
\end{itemize}
```### LaTeX snippet for your resume

```latex
% Projects section entry (works in most resume templates)
\section{Projects}
\noindent\textbf{AI-Powered Data Science Mock Interviewer} \hfill \textit{Python, Flask, LangChain, Groq, FAISS, MongoDB, ElevenLabs} \\
\href{https://github.com/agnihotri-anxh/AI-Powered-Data-Science-Mock-Interviewer}{github.com/agnihotri-anxh/AI-Powered-Data-Science-Mock-Interviewer} \hfill \textit{2025}
\begin{itemize}\itemsep 0.2em
  \item Built a mock interviewer that generates contextual questions from “The Hundred-Page Machine Learning Book” and evaluates answers with LLMs.
  \item Implemented retrieval-augmented generation using sentence-transformer embeddings with FAISS and a Groq-hosted model for question generation.
  \item Added relevance checking and a final comprehensive evaluation after 10 Q\&A pairs (summary, strengths, improvements, score).
  \item Developed user auth with MongoDB (signup/login, session management) and input validation for topic/answer quality.
  \item Integrated ElevenLabs TTS to synthesize questions; created a clean Tailwind UI for interview flow.
  \item Automated PDF ingestion pipeline to build a reusable vector knowledge base.
\end{itemize}
```

If your template uses a “Work/Project” entry format with dates/location, you can wrap it like:

```latex
\textbf{AI-Powered Data Science Mock Interviewer} \hfill \textit{Remote \quad 2025}\\
\textit{Python, Flask, LangChain, Groq, FAISS, MongoDB, ElevenLabs} \hfill
\href{https://github.com/agnihotri-anxh/AI-Powered-Data-Science-Mock-Interviewer}{GitHub}
\begin{itemize}\itemsep 0.2em
  \item Built RAG-driven interviewer; generated questions from curated ML book context; delivered final scored feedback.
  \item Implemented FAISS vector store, relevance checks, MongoDB auth, and TTS for accessibility.
\end{itemize}
```