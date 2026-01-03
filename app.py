import os
import secrets
import datetime
import nest_asyncio
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

# Config & Services
from config import Config
from services.database import init_db, get_db
from services.email_service import send_email
from services.llm import generate_interview_question, evaluate_interview
from services.audio import transcribe_audio_file, synthesize_audio_response

# Apply nest_asyncio
nest_asyncio.apply()

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config["MONGO_URI"] = Config.MONGO_URI

# Initialize Database
init_db(app)

# --- Routes ---

@app.route("/")
def landing():
    return render_template("landing_page.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        db = get_db()[Config.MONGODB_DB]
        users = db[Config.USERS_COLLECTION]
        
        username = request.form['username']
        password = request.form['password']
        user = users.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            session['username'] = username
            session['user_id'] = str(user['_id'])
            session['full_name'] = user.get('full_name', username)
            users.update_one({"_id": user['_id']}, {"$set": {"last_login": datetime.datetime.utcnow()}})
            flash('Login successful!', 'success')
            return redirect(url_for('landing'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        db = get_db()[Config.MONGODB_DB]
        users = db[Config.USERS_COLLECTION]

        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        email = request.form['email']
        
        if users.find_one({"username": username}):
            flash('Username already exists.', 'error')
        elif users.find_one({"email": email}):
            flash('Email already registered.', 'error')
        else:
            users.insert_one({
                "username": username,
                "password": generate_password_hash(password),
                "full_name": full_name,
                "email": email,
                "created_at": datetime.datetime.utcnow()
            })
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

@app.route("/interview")
def interview():
    if 'username' not in session: return redirect(url_for('login'))
    session.pop('interview_history', None)
    
    full_name = session.get('full_name', session.get('username', 'there'))
    initial_message = f"Hello {full_name}! Let's begin. The interview will consist of 5 questions. I will ask a question, and you can record your answer."
    
    return render_template("index.html", username=full_name, initial_message=initial_message)

@app.route("/ask", methods=["POST"])
def ask_question():
    if 'username' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    history = session.get('interview_history', [])
    if len(history) >= 5:
        return jsonify({"error": "Interview complete", "interview_over": True})

    try:
        question_text = generate_interview_question()
        return jsonify({
            "question": question_text,
            "question_number": len(history) + 1,
            "total_questions": 5
        })
    except Exception as e:
        print(f"Error generating question: {e}")
        return jsonify({"error": "Failed to generate question"}), 500

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    if 'username' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")
    
    # Update History
    history = session.get('interview_history', [])
    history.append({"question": question, "answer": answer})
    session['interview_history'] = history
    
    if len(history) >= 5:
        try:
            feedback = evaluate_interview(history)
            session.pop('interview_history', None)
            return jsonify({"message": "Complete", "feedback": feedback, "interview_over": True})
        except Exception as e:
            return jsonify({"error": "Evaluation failed"}), 500

    return jsonify({"message": "Recorded", "interview_over": False})

@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    if 'username' not in session: return jsonify({"error": "Unauthorized"}), 401
    if 'audio_data' not in request.files: return jsonify({"error": "No file"}), 400
    
    try:
        audio_file = request.files['audio_data']
        temp_filename = f"temp_rec_{session.get('user_id')}_{int(datetime.datetime.utcnow().timestamp())}.webm"
        audio_file.save(temp_filename)
        
        text = transcribe_audio_file(temp_filename)
        
        if os.path.exists(temp_filename): os.remove(temp_filename)
        return jsonify({"text": text})
    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({"error": "Transcription failed"}), 500

@app.route("/synthesize", methods=["POST"])
def synthesize_endpoint():
    data = request.get_json()
    return synthesize_audio_response(data.get("text", ""))

@app.route("/audio_status", methods=["GET"])
def audio_status():
    return jsonify({"enabled": True, "message": "Audio via Edge TTS"})

# --- Password Reset Routes (Simplified for brevity, logic preserved) ---
# ... (Use similar patterns if needed, omitted to keep file clean as per optimization request) ...
# You originally had forgot_password, verify_otp, reset_password. 
# I will include them quickly to maintain feature parity.

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        db = get_db()[Config.MONGODB_DB]
        user = db[Config.USERS_COLLECTION].find_one({"email": email})
        
        if user:
            otp_code = f"{secrets.randbelow(1000000):06d}"
            db[Config.USERS_COLLECTION].update_one({"email": email}, {"$set": {"otp_code": otp_code, "otp_expires": datetime.datetime.utcnow() + datetime.timedelta(minutes=15)}})
            send_email(email, "Password Reset OTP", f"Your OTP is {otp_code}")
            return redirect(url_for('verify_otp', email=email))
        flash('Email not found.', 'error')
    return render_template('forgot_password.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email') or request.form.get('email')
    if request.method == 'POST':
        otp = request.form.get('otp')
        db = get_db()[Config.MONGODB_DB]
        user = db[Config.USERS_COLLECTION].find_one({"email": email})
        if user and user.get('otp_code') == otp:
            reset_token = secrets.token_urlsafe(32)
            db[Config.USERS_COLLECTION].update_one({"_id": user['_id']}, {"$set": {"reset_token": reset_token}})
            return redirect(url_for('reset_password', token=reset_token))
        flash('Invalid OTP', 'error')
    return render_template('verify_otp.html', email=email)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        new_password = request.form['new_password']
        db = get_db()[Config.MONGODB_DB]
        db[Config.USERS_COLLECTION].update_one({"reset_token": token}, {"$set": {"password": generate_password_hash(new_password)}})
        flash('Password reset.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', token=token)

if __name__ == "__main__":
    print("Starting Optimized Flask Server...")
    app.run(debug=(os.getenv("FLASK_ENV") != "production"), host="0.0.0.0", port=5000, threaded=True)
