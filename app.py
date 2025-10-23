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

# --- INITIALIZATION ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "a-super-secret-key-that-should-be-changed")

# Memory optimization: Force garbage collection
gc.collect()

# Print initial memory usage
def print_memory_usage(stage=""):
    memory_info = psutil.virtual_memory()
    print(f"🧠 Memory Usage {stage}: {memory_info.percent:.1f}% ({memory_info.used / 1024 / 1024:.1f} MB / {memory_info.total / 1024 / 1024:.1f} MB)")

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


# --- API & MODEL CONFIGURATION ---
groq_api_key = os.getenv("GROQ_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY must be set in the .env file.")

# Initialize LLM (ElevenLabs is optional for memory optimization)
llm = ChatGroq(model_name="mixtral-8x7b-32768")
elevenlabs_client = None

if elevenlabs_api_key:
    elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
    print("✅ ElevenLabs client initialized.")
else:
    print("⚠️ ElevenLabs API key not found. Audio features will be disabled.")


# --- KNOWLEDGE BASE LOADING ---
try:
    print("🔄 Loading knowledge base...")
    vectorstore = DataScienceKnowledgeExtractor.load_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print_memory_usage("(After Knowledge Base Load)")
    gc.collect()  # Force garbage collection after loading
except FileNotFoundError:
    print("\n[ERROR] Knowledge base not found. Please run 'python run_extraction.py' first.\n")
    exit()


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


question_generation_chain = (
    {"context": retriever, "topic": RunnablePassthrough()} | question_prompt | llm | StrOutputParser()
)

relevance_check_chain = relevance_prompt | llm | StrOutputParser()

final_evaluation_chain = final_evaluation_prompt | llm | StrOutputParser()


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
    
    data = request.get_json()
    topic = data.get("topic")
    if not topic: return jsonify({"error": "Topic not provided"}), 400

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
        
        result = question_generation_chain.invoke(topic)
        return jsonify({"question": result})
    except Exception as e:
        print(f"Error in /ask: {e}")
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
    print_memory_usage("(Final)")
    print("Access the application at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, threaded=True)

