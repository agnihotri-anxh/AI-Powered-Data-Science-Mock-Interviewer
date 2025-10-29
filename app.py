import os
import gc
import psutil
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for, flash
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from elevenlabs.client import ElevenLabs
from Data_Ingestion import DataScienceKnowledgeExtractor
from flask_pymongo import PyMongo
import datetime
from pymongo.errors import ConnectionFailure
import re
import secrets
import smtplib
from email.message import EmailMessage

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "a-super-secret-key-that-should-be-changed")

gc.collect()

def print_memory_usage(stage=""):
    """Simplified memory monitoring"""
    memory_info = psutil.virtual_memory()
    print(f"🧠 Memory {stage}: {memory_info.percent:.1f}% ({memory_info.used / 1024 / 1024:.1f} MB)")
    
    # Force garbage collection if memory usage is high
    if memory_info.percent > 70:
        collected = gc.collect()
        print(f"🗑️ GC: {collected} objects")

print_memory_usage("(Startup)")

# --- MONGODB CONFIGURATION ---
DB_NAME = os.getenv("MONGODB_DB", "AI-Interviewer-DB")
USERS_COLLECTION_NAME = os.getenv("MONGODB_USERS_COLLECTION", "users")

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
if not app.config["MONGO_URI"]:
    raise ValueError("MONGO_URI not found in .env file. Please add your MongoDB connection string.")

try:
    mongo = PyMongo(app)
    mongo.cx.admin.command('ismaster')
    print("✅ MongoDB connection successful.")
except ConnectionFailure as e:
    print("\n❌ MongoDB connection failed. Please check your MONGO_URI and network access rules.")
    print(f"Error details: {e}\n")
    exit()
# --- EMAIL CONFIGURATION ---
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER or "")

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASSWORD and MAIL_FROM):
        print("⚠️ Email not configured; skipping send.")
        return False
    try:
        msg = EmailMessage()
        msg["From"] = MAIL_FROM
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"📧 Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


# --- API & MODEL CONFIGURATION ---
groq_api_key = os.getenv("GROQ_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY must be set in the .env file.")

# Lazy loading for LLM and ElevenLabs
llm = None
elevenlabs_client = None

def get_llm():
    """Lazy load the LLM only when needed"""
    global llm
    if llm is None:
        print("🔄 Loading LLM model...")
        # Use a lighter, faster model to reduce latency and memory on Render
        llm = ChatGroq(model_name="llama-3.1-8b-instant")
        print("✅ LLM model loaded")
    return llm

def get_elevenlabs_client():
    """Lazy load ElevenLabs client only when needed"""
    global elevenlabs_client
    if elevenlabs_client is None and elevenlabs_api_key:
        print("🔄 Loading ElevenLabs client...")
        elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
        print("✅ ElevenLabs client loaded")
    return elevenlabs_client


# --- KNOWLEDGE BASE CONFIGURATION ---
# Lazy loading - knowledge base will be loaded on first use
vectorstore = None
retriever = None

def get_knowledge_base():
    """Lazy load the knowledge base only when needed. If loading fails, return (None, None)."""
    global vectorstore, retriever
    
    if vectorstore is None:
        try:
            print("🔄 Loading knowledge base...")
            print_memory_usage("(Before KB Load)")
            
            vectorstore = DataScienceKnowledgeExtractor.load_knowledge_base()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            
            print_memory_usage("(After KB Load)")
            gc.collect()
            print("✅ Knowledge base loaded")
        except FileNotFoundError:
            print("\n⚠️ Knowledge base directory not found. Continuing without RAG context.\n")
            vectorstore, retriever = None, None
        except Exception as e:
            print(f"\n⚠️ Knowledge base load failed: {e}. Continuing without RAG context.\n")
            gc.collect()
            vectorstore, retriever = None, None
    
    return vectorstore, retriever


# --- PROMPT TEMPLATES & LANGCHAIN CHAINS ---
question_generation_template = """
You are an expert AI Data Science interviewer using the 'Hundred-Page Machine Learning Book' as your knowledge base.
Your persona is professional, encouraging, and focused. Your goal is to assess the user's technical knowledge.

**Instructions:**
1. Based on the provided context, generate a single, insightful interview question about the topic: '{topic}'.
2. The question should be practical and test real-world understanding, not just rote memorization.
3. Do not greet the user or add conversational fluff. Only return the interview question itself.
4. **Keep the question concise and to the point (ideally under 50 words).**

**Context from the knowledge base:**
{context}
**Interview Question:**
"""
question_prompt = ChatPromptTemplate.from_template(question_generation_template)

# New template for checking answer relevance
relevance_check_template = """
You are an AI assistant helping to evaluate if a user's answer is relevant to a data science interview question.

**Question:** {question}
**User's Answer:** {answer}

**Instructions:**
1. Determine if the answer is relevant to the question asked
2. Consider if the answer demonstrates knowledge of data science, machine learning, statistics, or related technical concepts
3. Look for technical terms, concepts, or explanations that relate to the question

**Response Format:**
- If relevant: "RELEVANT"
- If not relevant: "NOT_RELEVANT: [brief explanation of why it's not relevant]"

**Examples of NOT_RELEVANT responses:**
- Personal stories unrelated to the technical question
- General conversation or greetings
- Completely different topics (cooking, sports, etc.)
- Nonsensical text or gibberish
- Questions back to the interviewer instead of answers

**Examples of RELEVANT responses:**
- Technical explanations of concepts
- Code examples or algorithms
- Mathematical or statistical explanations
- Industry experience related to the question
- Even partial or incorrect technical answers are considered relevant

**Your assessment:**
"""
relevance_prompt = ChatPromptTemplate.from_template(relevance_check_template)

final_evaluation_template = """
You are an expert AI Data Science hiring manager providing final, comprehensive feedback on a mock interview.
Your tone should be professional, constructive, and encouraging.
**Interview Transcript:**
{interview_transcript}
**Instructions:**
Based on the entire conversation, provide a single, detailed evaluation in a professional format. If the user asked clarifying questions, acknowledge this positively as a sign of engagement. If they challenged a premise, evaluate their reasoning. The goal is to provide feedback on their technical communication and problem-solving skills as demonstrated in the transcript.
Your feedback must include the following sections:
1.  **Overall Summary:** A brief, two-sentence summary of the candidate's performance and grasp of the topics.
2.  **Key Strengths:** List 2-3 specific strengths the candidate demonstrated, referencing their answers (e.g., "Showed strong foundational knowledge when explaining...").
3.  **Areas for Improvement:** List 2-3 specific, actionable areas where the candidate could improve (e.g., "Could provide more detail on the practical trade-offs of...").
4.  **Overall Score:** Provide a score out of 10 (e.g., 7.5/10).
5.  **Final Recommendation:** A concluding, encouraging sentence for the candidate.
"""
final_evaluation_prompt = ChatPromptTemplate.from_template(final_evaluation_template)


def create_question_generation_chain():
    """Create the question generation chain with optional retriever and lazy-loaded LLM.
    Falls back to no-context generation if retriever is unavailable.
    """
    _, _retriever = get_knowledge_base()
    llm = get_llm()
    if _retriever is not None:
        return (
            {"context": _retriever, "topic": RunnablePassthrough()} | question_prompt | llm | StrOutputParser()
        )
    # Fallback: no RAG context
    return (
        {"context": "", "topic": RunnablePassthrough()} | question_prompt | llm | StrOutputParser()
    )

def create_relevance_check_chain():
    """Create the relevance check chain with lazy-loaded LLM"""
    llm = get_llm()
    return relevance_prompt | llm | StrOutputParser()

def create_final_evaluation_chain():
    """Create the final evaluation chain with lazy-loaded LLM"""
    llm = get_llm()
    return final_evaluation_prompt | llm | StrOutputParser()


# --- AUTHENTICATION ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        db = mongo.cx[DB_NAME]
        users_collection = db[USERS_COLLECTION_NAME]
        
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            session['username'] = username
            session['user_id'] = str(user['_id'])
            session['full_name'] = user.get('full_name', username)
            
            # Update last login
            users_collection.update_one(
                {"_id": user['_id']}, 
                {"$set": {"last_login": datetime.datetime.utcnow()}}
            )
            
            flash('Login successful!', 'success')
            return redirect(url_for('landing'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        db = mongo.cx[DB_NAME]
        users_collection = db[USERS_COLLECTION_NAME]

        # Get form data
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        email = request.form['email']
        
        # Validation
        existing_user = users_collection.find_one({"username": username})
        existing_email = users_collection.find_one({"email": email})

        if existing_user:
            flash('Username already exists.', 'error')
        elif existing_email:
            flash('Email already registered.', 'error')
        elif not username or not password or not full_name or not email:
            flash('All required fields must be filled.', 'error')
        else:
            # Create user document with simplified fields
            user_data = {
                "username": username,
                "password": generate_password_hash(password),
                "full_name": full_name,
                "email": email,
                "created_at": datetime.datetime.utcnow(),
                "last_login": None
            }
            
            users_collection.insert_one(user_data)
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        db = mongo.cx[DB_NAME]
        users_collection = db[USERS_COLLECTION_NAME]
        
        email = request.form['email']
        user = users_collection.find_one({"email": email})
        
        if user:
            # Generate a 6-digit OTP
            otp_code = f"{secrets.randbelow(1000000):06d}"
            users_collection.update_one(
                {"email": email},
                {"$set": {
                    "otp_code": otp_code,
                    "otp_expires": datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
                }}
            )
            # Send email with OTP
            email_sent = send_email(
                email,
                "Your password reset OTP",
                f"Your OTP code is {otp_code}. It will expire in 15 minutes."
            )
            if email_sent:
                flash('An OTP has been sent to your email. Please enter it below to continue.', 'success')
            else:
                flash('Failed to send OTP email. Please contact support or try again.', 'error')
            return redirect(url_for('verify_otp', email=email))
        else:
            flash('Email not found.', 'error')
    
    return render_template('forgot_password.html')
@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    db = mongo.cx[DB_NAME]
    users_collection = db[USERS_COLLECTION_NAME]

    email = request.args.get('email') or request.form.get('email')
    if request.method == 'POST':
        otp = request.form.get('otp')
        if not email or not otp:
            flash('Email and OTP are required.', 'error')
        else:
            user = users_collection.find_one({"email": email})
            now = datetime.datetime.utcnow()
            if user and user.get('otp_code') == otp and user.get('otp_expires') and user['otp_expires'] > now:
                reset_token = secrets.token_urlsafe(32)
                users_collection.update_one(
                    {"_id": user['_id']},
                    {"$set": {
                        "reset_token": reset_token,
                        "reset_token_expires": now + datetime.timedelta(hours=1)
                    }, "$unset": {"otp_code": "", "otp_expires": ""}}
                )
                flash('OTP verified. You can now reset your password.', 'success')
                return redirect(url_for('reset_password', token=reset_token))
            else:
                flash('Invalid or expired OTP.', 'error')
    return render_template('verify_otp.html', email=email)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    db = mongo.cx[DB_NAME]
    users_collection = db[USERS_COLLECTION_NAME]
    
    user = users_collection.find_one({
        "reset_token": token,
        "reset_token_expires": {"$gt": datetime.datetime.utcnow()}
    })
    
    if not user:
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
        elif len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
        else:
            # Update password and clear reset token
            users_collection.update_one(
                {"_id": user['_id']},
                {"$set": {
                    "password": generate_password_hash(new_password),
                    "reset_token": None,
                    "reset_token_expires": None
                }}
            )
            flash('Password has been reset successfully. Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)


# --- CORE APPLICATION ROUTES ---

@app.route("/")
def landing():
    return render_template("landing_page.html")

@app.route("/interview")
def interview():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    session.pop('interview_history', None)
    
    # Get user information for personalized introduction
    full_name = session.get('full_name', session.get('username', 'there'))
    
    # Create simple personalized introduction
    initial_message = f"Hello {full_name}! Let's begin. The interview will consist of 10 questions. Please enter your first data science topic below."
    
    return render_template("index.html", username=full_name, initial_message=initial_message)


@app.route("/ask", methods=["POST"])
def ask_question():
    if 'username' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    # Accept topic from JSON, form, or query to be robust to different clients
    data = None
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    topic = (
        (data or {}).get("topic")
        or request.form.get("topic")
        or request.args.get("topic")
    )
    if not topic:
        return jsonify({"error": "Topic not provided"}), 400

    # Normalize and support phrases like "from the same" to reuse last topic
    topic_normalized_input = str(topic).strip().strip('"\'').lower()
    reuse_phrases = {"same", "from the same", "same topic", "continue", "more", "another", "next"}
    if topic_normalized_input in reuse_phrases:
        last_topic = session.get('last_topic')
        if last_topic:
            topic = last_topic
        else:
            return jsonify({"error": "No previous topic to continue from. Please enter a topic like 'Neural Networks'."})
    else:
        # Store last concrete topic
        session['last_topic'] = topic

    # Topic validation
    topic_lower = topic.lower().strip()
    topic_normalized = re.sub(r"\s+", " ", topic_lower)
    
    # Quick checks
    if len(topic_normalized) < 3:
        return jsonify({"error": "Please provide a more specific data science topic (e.g., 'Machine Learning', 'Neural Networks', 'Cross Validation')."})
    
    # Check for irrelevant phrases (whole-word match to avoid substrings like 'hi' in 'machine')
    irrelevant = {"hi", "hello", "hey", "thanks", "bye", "weather", "food", "sports", "music", "movie", "politics"}
    words_in_topic = set(re.findall(r"[a-z]+", topic_normalized))
    if any(word in irrelevant for word in words_in_topic):
        return jsonify({"error": f"'{topic}' doesn't seem to be a data science topic. Please enter a relevant topic like 'Machine Learning', 'Neural Networks', 'Cross Validation', or 'Feature Engineering'."})

    try:
        if 'interview_history' not in session:
            session['interview_history'] = []
        
        # Primary path: use chain (with retriever if available)
        try:
            question_generation_chain = create_question_generation_chain()
            result = question_generation_chain.invoke(topic)
            if not result or not str(result).strip():
                raise ValueError("Empty question from chain")
            return jsonify({"question": str(result).strip()})
        except Exception as chain_err:
            print(f"/ask chain failed: {chain_err}")
            # Fallback: direct prompt without RAG/chain
            try:
                llm_instance = get_llm()
                prompt = (
                    "You are an expert data science interviewer. "
                    f"Generate one concise, practical interview question about '{topic}'. "
                    "Return only the question."
                )
                result = llm_instance.invoke(prompt)
                # Extract plain text from LLM result (handles AIMessage objects)
                text = None
                try:
                    # LangChain messages usually have `.content`
                    text = getattr(result, "content", None)
                except Exception:
                    text = None
                if not text:
                    text = str(result) if result is not None else ""
                if not text.strip():
                    raise ValueError("Empty question from LLM fallback")
                return jsonify({"question": text.strip()})
            except Exception as llm_err:
                print(f"/ask LLM fallback failed: {llm_err}")
                # Final static fallback so UI continues
                generic = f"What are the key concepts and trade-offs involved in {topic}?"
                return jsonify({"question": generic})
    except Exception as e:
        print(f"Error in /ask: {e}")
        gc.collect()
        return jsonify({"error": "Failed to generate question"}), 500

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    if 'username' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")
    if not question or not answer:
        return jsonify({"error": "Question or answer not provided"}), 400
        
    # Input validation
    if len(answer.strip()) < 3:
        return jsonify({"message": "Please provide a more detailed answer. If you don't know, you can say 'I don't know' and we'll move on.", "interview_over": False})
    
    if len(set(answer.strip())) < 3 and len(answer.strip()) > 5:
        return jsonify({"message": "It looks like your response might not be a proper answer. Please try to provide a relevant response to the question, or say 'I don't know' if you're unsure.", "interview_over": False})
    
    # Check if user is skipping
    normalized_answer = re.sub(r'[^\w\s]', '', answer.lower().strip())
    skip_phrases = ["i dont know", "no idea", "skip", "pass", "idk", "not sure"]
    user_is_skipping = any(phrase in normalized_answer for phrase in skip_phrases)
    
    # Check if answer is relevant to the question (only if not skipping)
    answer_is_relevant = True
    relevance_feedback = ""
    
    if not user_is_skipping and len(answer.strip()) > 10:
        try:
            relevance_check_chain = create_relevance_check_chain()
            relevance_result = relevance_check_chain.invoke({"question": question, "answer": answer})
            if relevance_result.startswith("NOT_RELEVANT"):
                answer_is_relevant = False
                relevance_feedback = relevance_result.replace("NOT_RELEVANT:", "").strip()
        except Exception as e:
            # If relevance check fails, assume answer is relevant to avoid blocking valid responses
            pass
    
    history = session.get('interview_history', [])
    if user_is_skipping:
        history.append({"question": question, "answer": "(User indicated they did not know the answer.)"})
    elif not answer_is_relevant:
        history.append({"question": question, "answer": f"(User provided an off-topic response: {answer})"})
    else:
        history.append({"question": question, "answer": answer})
    session['interview_history'] = history

    if len(session['interview_history']) >= 10:
        transcript = ""
        for i, item in enumerate(history):
            transcript += f"Question {i+1}: {item['question']}\nAnswer {i+1}: {item['answer']}\n\n"
        
        try:
            final_evaluation_chain = create_final_evaluation_chain()
            final_feedback = final_evaluation_chain.invoke({"interview_transcript": transcript})
            session.pop('interview_history', None)
            return jsonify({"feedback": final_feedback, "interview_over": True})
        except Exception as e:
            return jsonify({"error": "Failed to generate final feedback"}), 500

    if user_is_skipping:
        follow_up_message = "That's perfectly fine. Let's move on. What's the next topic you'd like to cover?"
    elif not answer_is_relevant:
        follow_up_message = f"I notice your response doesn't seem to relate to the question about data science. {relevance_feedback} Please try to answer the technical question, or if you're unsure, you can say 'I don't know' and we'll move to the next topic."
    else:
        follow_up_message = "Okay, thank you for your answer. What is the next topic you would like to discuss?"
    
    return jsonify({"message": follow_up_message, "interview_over": False})


@app.route("/synthesize", methods=["POST"])
def synthesize():
    if 'username' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    elevenlabs_client = get_elevenlabs_client()
    if not elevenlabs_client: return jsonify({"error": "Audio features disabled"}), 503
    
    data = request.get_json()
    text = data.get("text")
    if not text: return jsonify({"error": "Text not provided"}), 400
    try:
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            optimize_streaming_latency="0",
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_multilingual_v2",
        )
        return Response(audio_stream, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Error in /synthesize: {e}")
        return jsonify({"error": "Failed to synthesize audio"}), 500


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting Flask server...")
    print_memory_usage("(Startup)")
    print("Access the application at http://127.0.0.1:5000")
    
    # Use production settings for Render deployment
    debug_mode = os.getenv("FLASK_ENV") != "production"
    port = int(os.getenv("PORT", 5000))
    host = "0.0.0.0" if os.getenv("FLASK_ENV") == "production" else "127.0.0.1"
    
    app.run(debug=debug_mode, host=host, port=port, threaded=True)

