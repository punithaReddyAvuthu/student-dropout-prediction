import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
import random
import os

# Initialize components
print("Initializing Chatbot Engine...")
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

# Global generator (lazy loaded)
generator = None

def load_generator():
    """Lazy loads the transformer model only when needed."""
    global generator
    if generator is None:
        try:
            print("⏳ Loading Transformers pipeline (distilgpt2)... This may take a moment.")
            # use device=-1 for CPU
            generator = pipeline("text-generation", model="distilgpt2", device=-1)
            print("✅ Transformers pipeline loaded.")
        except Exception as e:
            print(f"Chatbot Init Error (Transformers): {e}")
            generator = "FAILED" # Mark as failed to avoid repeated attempts
    return generator

# Predefined empathetic responses for high stress
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
    """
    Analyzes the query for stress and intent, then generates a response.
    """
    # 1. Stress Detection (Sentiment) - Use a more sensitive threshold
    is_stressed = False
    if sid:
        sentiment = sid.polarity_scores(user_query)
        # -0.05 is the standard threshold for "negative" in VADER
        is_stressed = sentiment['compound'] <= -0.1
    
    # 2. Intent Detection (Keyword Substring matching is more robust)
    query_clean = user_query.lower()
    
    intent = "general"
    
    # Academic keywords
    if any(k in query_clean for k in ["exam", "cgpa", "study", "studying", "plan", "marks", "backlog", "assignment"]):
        intent = "academic"
    
    # Stress keywords (Check specifically for stress words)
    if any(k in query_clean for k in ["stress", "anxi", "sad", "fail", "scared", "tired", "quit", "overwhelmed", "help"]):
        intent = "stress"

    # 3. Generate Response
    if intent == "stress" or is_stressed:
        base_msg = random.choice(STRESS_RESPONSES)
    elif intent == "academic":
        base_msg = random.choice(ACADEMIC_ADVICE)
    else:
        # Use AI generation for general queries
        gen = load_generator()
        if gen and gen != "FAILED":
            try:
                # Prompt engineering to keep GPT-2 focused on student counseling
                prompt = f"Student says: {user_query}\nCounselor advice:"
                ai_out = gen(prompt, max_length=50, num_return_sequences=1, truncation=True)
                # Clean up the response
                raw_text = ai_out[0]['generated_text']
                base_msg = raw_text.split("Counselor advice:")[-1].strip()
                if not base_msg:
                    base_msg = "I'm here to support you. Tell me more about what's on your mind."
            except:
                base_msg = "I understand. Let's talk more about how I can help you with your studies or well-being."
        else:
            base_msg = "I'm here to support you. How can I help with your academic journey today?"

    return base_msg

if __name__ == "__main__":
    # Test cases
    print(f"Test 1 (Stress): {get_chatbot_response('I feel very stressed about my exams')}")
    print(f"Test 2 (Academic): {get_chatbot_response('How can I improve my CGPA?')}")
    print(f"Test 3 (General): {get_chatbot_response('Hello bot')}")
