from flask_pymongo import PyMongo
from pymongo.errors import ConnectionFailure

mongo = PyMongo()

def init_db(app):
    """Initialize MongoDB with the Flask app"""
    try:
        mongo.init_app(app)
        # Verify connection
        # Note: In newer PyMongo/Flask-PyMongo versions, we might need a request context or 
        # direct access to check connection immediately, but lazy connection is standard.
        print(" MongoDB initialized.")
    except Exception as e:
        print(f" MongoDB initialization failed: {e}")

def get_db():
    return mongo.cx
