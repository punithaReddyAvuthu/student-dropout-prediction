import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import os

# Initialize components
print("Initializing Optimized Chatbot Engine...")
try:
    # Ensure NLTK resources are available
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    
    # Try to load spaCy model, but don't crash if missing
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"⚠️ Warning: spaCy 'en_core_web_sm' not found. Intent detection will be limited. {e}")
        nlp = None
        
    sid = SentimentIntensityAnalyzer()
    print("✅ Chatbot components initialized.")
except Exception as e:
    print(f"Chatbot Init Error (NLTK/spaCy): {e}")
    nlp = None
    sid = None

# Cloud Optimization: Removing Transformers for speed and stability
# We use a diverse set of canned responses for general queries

GENERAL_RESPONSES = [
    "I'm your AI counselor, here to help you navigate your academic journey. Is there a specific subject or stress factor you'd like to discuss?",
    "Hello! I'm here to support your success. Would you like to talk about study techniques, stress management, or perhaps your risk level?",
    "I understand. Your well-being is as important as your grades. How can I assist you today?",
    "That's interesting. Many students find that talking through their concerns is the first step toward improvement. Tell me more.",
    "I'm listening. Whether it's about your CGPA, your sleep schedule, or just general anxiety, I'm here for you.",
    "I'm specialized in helping students manage academic risk and stress. What's on your mind right now?"
]

GREETINGS = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
WHO_ARE_YOU = ["who are you", "what can you do", "help me", "your name", "purpose"]

STRESS_RESPONSES = [
    "I understand that things feel overwhelming right now. Remember that you don't have to carry this alone. Would you like to break down your tasks into a small study plan?",
    "It's completely normal to feel stressed. You've handled difficult situations before, and you can handle this too. Let's focus on one step at a time.",
    "I hear you. Taking a 10-minute break for deep breathing (the 4-7-8 technique) can sometimes help clear the mind. Should we look at your current academic priorities?",
    "Stress is often a sign that you care deeply about your goals. Let's try to channel that energy into a structured schedule. What is the one thing bothering you the most today?",
    "When things feel out of control, try the 5-4-3-2-1 technique: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you can taste. Does that help ground you?",
    "You are more than just your grades. Your well-being is the foundation of your success. Have you had enough sleep or a proper meal today?",
    "It's okay to take a 'zero-pressure' day to recharge. Sometimes stepping back actually helps you move forward faster later.",
    "If the workload feels like a mountain, let's just focus on the first few steps. What is the simplest task on your list that we can finish in 15 minutes?",
    "I'm here for you. Many students feel this way during the semester. You're doing better than you think!"
]

ACADEMIC_ADVICE = [
    "To improve your CGPA, consistency is key. Try setting aside 2 dedicated hours each day for your core subjects.",
    "For exam anxiety, practice active recall and spaced repetition. It builds much more confidence than just re-reading notes.",
    "If you're struggling with participation, try asking just one question in every class. It makes a big difference in internal marks!",
    "Try the Pomodoro Technique: Study for 25 minutes, then take a 5-minute break. It keeps your brain fresh and prevents burnout.",
    "Don't hesitate to visit your professors during office hours. They appreciate students who show interest and can clarify difficult concepts early.",
    "Try 'Feynman Technique': Explain what you've learned to an imaginary student. If you can't explain it simply, you don't understand it yet.",
    "Your environment matters. Try a 'digital detox'—put your phone in another room while studying to maximize your focus.",
    "Peer study groups can be very effective if you stay on task. Explaining concepts to others is one of the best ways to learn.",
    "Prioritize your 'backlogs' by tackling the most difficult subject first when your energy is highest in the morning."
]

def get_chatbot_response(user_query):
    query_clean = user_query.lower().strip()
    
    # Check for basic greetings
    if any(greet in query_clean for greet in GREETINGS):
        return random.choice(GENERAL_RESPONSES)

    # Check for identity questions
    if any(q in query_clean for q in WHO_ARE_YOU):
        return "I am an AI-powered Student Counselor. I can predict your dropout risk and offer advice on managing stress and improving your grades."

    # 1. Stress Detection (Sentiment)
    is_stressed = False
    if sid:
        sentiment = sid.polarity_scores(user_query)
        is_stressed = sentiment['compound'] <= -0.1
    
    # 2. Intent Detection
    intent = "general"
    if any(k in query_clean for k in ["exam", "cgpa", "study", "studying", "plan", "marks", "backlog", "assignment"]):
        intent = "academic"
    if any(k in query_clean for k in ["stress", "anxi", "sad", "fail", "scared", "tired", "quit", "overwhelmed", "help"]):
        intent = "stress"

    # 3. Generate Response
    if intent == "stress" or is_stressed:
        return random.choice(STRESS_RESPONSES)
    elif intent == "academic":
        return random.choice(ACADEMIC_ADVICE)
    
    # Fallback to general advice
    return random.choice(GENERAL_RESPONSES)
