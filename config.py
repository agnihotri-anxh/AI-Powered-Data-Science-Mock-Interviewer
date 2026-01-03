import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "a-super-secret-key-that-should-be-changed")
    MONGO_URI = os.getenv("MONGO_URI")
    MONGODB_DB = os.getenv("MONGODB_DB", "AI-Interviewer-DB")
    USERS_COLLECTION = os.getenv("MONGODB_USERS_COLLECTION", "users")
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # SMTP Settings
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER or "")
