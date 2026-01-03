import os
import gc
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from Data_Ingestion import DataScienceKnowledgeExtractor
from config import Config

# --- Globals ---
llm = None
vectorstore = None
retriever = None

# --- Prompts ---
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

**Your assessment:**
"""

final_evaluation_template = """
You are an expert AI Data Science hiring manager providing final, comprehensive feedback on a mock interview.
Your tone should be professional, constructive, and encouraging.
**Interview Transcript:**
{interview_transcript}
**Instructions:**
Based on the entire conversation, provide a single, detailed evaluation in a professional format.
Your feedback must include the following sections:
1.  **Overall Summary:** A brief, two-sentence summary of the candidate's performance.
2.  **Key Strengths:** List 2-3 specific strengths.
3.  **Areas for Improvement:** List 2-3 specific, actionable areas where the candidate could improve.
4.  **Overall Score:** Provide a score out of 10 (e.g., 7.5/10).
5.  **Final Recommendation:** A concluding, encouraging sentence.
"""

# --- Functions ---

def get_llm():
    """Lazy load the LLM"""
    global llm
    if llm is None:
        if not Config.GROQ_API_KEY:
             raise ValueError("GROQ_API_KEY must be set in .env")
        print(" Loading LLM model...")
        llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=Config.GROQ_API_KEY)
        print(" LLM model loaded")
    return llm

def get_knowledge_base():
    """Lazy load knowledge base"""
    global vectorstore, retriever
    
    if vectorstore is None:
        try:
            print(" Loading knowledge base...")
            vectorstore = DataScienceKnowledgeExtractor.load_knowledge_base()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            gc.collect()
            print(" Knowledge base loaded")
        except Exception as e:
            print(f"\nKnowledge base load failed: {e}. Continuing without RAG context.\n")
            vectorstore, retriever = None, None
    
    return vectorstore, retriever

def generate_interview_question():
    """Generates a simple interview question"""
    llm_instance = get_llm()
    # Simplified prompt as per user request
    prompt_text = (
        "You are an expert Data Science Interviewer. "
        "Generate one unique, simple, and random interview question about a basic Data Science or Machine Learning concept. "
        "The question should be conceptual and ask for a short explanation. "
        "Start the question with 'Explain', 'What is', or 'Describe'. "
        "Do NOT ask for code or complex system design. "
        "Return ONLY the question text."
    )
    result = llm_instance.invoke(prompt_text)
    return getattr(result, "content", str(result)).strip()

def evaluate_interview(history):
    """Generates final feedback"""
    transcript = ""
    for i, item in enumerate(history):
        transcript += f"Question {i+1}: {item['question']}\nAnswer {i+1}: {item['answer']}\n\n"
        
    llm_instance = get_llm()
    chain = ChatPromptTemplate.from_template(final_evaluation_template) | llm_instance | StrOutputParser()
    return chain.invoke({"interview_transcript": transcript})
