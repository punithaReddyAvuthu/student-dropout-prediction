import nltk
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Expert Counseling Knowledge Base ---
# Using structured intents for high-accuracy local matching
COUNSELING_KNOWLEDGE = {
    "academic_struggle": {
        "patterns": ["low marks", "bad cgpa", "failing subjects", "cannot understand", "struggling with studies", "backlogs", "failed exam"],
        "responses": {
            "High": "I hear you. With your current risk level, we need a proactive strategy. First, identify your 'Red Zone' subjects. Focus on the last 5 years' question papers—80% of exams come from 20% of themes. Can we start by setting up a 2-hour daily deep-focus session for your hardest subject?",
            "Medium": "Struggling with grades is a common hurdle. Since you're in the middle ground, small shifts in study habits like 'Active Recall' will help more than just reading notes. Try explaining a topic to an imaginary student; if you can't explain it simply, you don't understand it yet.",
            "Low": "You're actually performing well overall! If you're feeling a struggle in one area, it might just be a temporary dip. Keep your current momentum but shift your 'weakest' subject to your highest energy time in the morning."
        }
    },
    "attendance_issue": {
        "patterns": ["missed classes", "low attendance", "class participation", "absent", "cannot attend"],
        "responses": {
            "High": "Attendance is your most critical lever right now. Missing classes often leads to a 'Information Gap' that makes the next class harder. Try to commit to 100% attendance for just the NEXT 5 DAYS. It builds the habit and shows professors you are serious.",
            "Medium": "Low attendance might be why your internal marks are fluctuating. Try to aim for at least 75%—the 'magic number' for most universities. If you can't attend, ensure you get the notes from a peer within 24 hours.",
            "Low": "Your attendance record is strong. If you've missed a few recently, don't let it slip. Remember, class participation often directly impacts your self-confidence score in the portal."
        }
    },
    "stress_overwhelmed": {
        "patterns": ["overwhelmed", "too much stress", "cannot cope", "anxiety", "scared", "pressured", "mental health", "stressed out"],
        "responses": {
            "High": "I'm genuinely sorry you're feeling this way. When stress hits a 'High' level, your brain's logic center shuts down. Try the 4-7-8 breathing technique now: inhale 4s, hold 7s, exhale 8s. Once calm, we will tackle just ONE tiny task. No more.",
            "Medium": "Stress is often just 'Future Anxiety' masquerading as work. Let's break your mountain into pebbles. What is the one thing you can finish in the next 15 minutes? Completing it will give you the dopamine hit you need to continue.",
            "Low": "Even high achievers feel stress! Don't let your success become a burden. Ensure you're getting 7 hours of sleep—your brain cleans its 'academic toxins' during deep sleep."
        }
    },
    "time_management": {
        "patterns": ["no time", "late submissions", "procrastinating", "wasting time", "study plan", "schedule", "too busy"],
        "responses": {
            "High": "We need a 'Survival Schedule.' Block out fixed times for sleep and meals first, then find three 45-minute blocks for study. Use the Pomodoro technique: 25 mins study, 5 mins break. It prevents the 'fatigue crash' that leads to dropout risk.",
            "Medium": "You have the time, you just need a better 'Filter.' Use the Eisenhower Matrix: Focus on things that are Important but not yet Urgent. This stops the last-minute panic that leads to late submissions.",
            "Low": "Your submission rate is good, but if you're feeling rushed, try 'Time Boxing.' Assign a fixed end-time for every task. It forces your brain to be efficient instead of perfect."
        }
    },
    "general_greeting": {
        "patterns": ["hi", "hello", "hey", "good morning", "how are you"],
        "responses": {
            "High": "Hello. I'm glad you're here. We have some work to do to secure your academic future, but we'll do it together. How can I help you take the first step today?",
            "Medium": "Hi! I'm your AI Counselor. You're doing okay, but there's room to grow. What academic or personal concern is on your mind?",
            "Low": "Hello! Great to see you. Your profile looks stable—how can I help you maintain this excellence or perhaps reach for a higher CGPA?"
        }
    }
}

class SemanticMatcher:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Build training set
        self.all_patterns = []
        self.pattern_to_intent = []
        for intent, data in self.kb.items():
            for pattern in data["patterns"]:
                self.all_patterns.append(pattern)
                self.pattern_to_intent.append(intent)
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_patterns)

    def get_best_intent(self, user_query):
        query_vec = self.vectorizer.transform([user_query.lower()])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = np.argmax(similarities)
        
        # Only return if we meet a minimum confidence threshold
        if similarities[best_idx] > 0.15:
            return self.pattern_to_intent[best_idx]
        return "fallback"

# Initialize components
print("Initializing Advanced Semantic Chatbot Engine...")
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()
matcher = SemanticMatcher(COUNSELING_KNOWLEDGE)
print("[SUCCESS] Advanced local brain online.")

def get_chatbot_response(user_query, risk_level="Low", username="Student"):
    query_clean = user_query.lower().strip()
    
    # 1. Detect Intent Semantically
    intent = matcher.get_best_intent(query_clean)
    
    # 2. Check Sentiment for Empathy Layer
    sentiment = sid.polarity_scores(query_clean)
    is_very_negative = sentiment['compound'] <= -0.5
    
    # 3. Generate Expert Response
    if intent in COUNSELING_KNOWLEDGE:
        response = COUNSELING_KNOWLEDGE[intent]["responses"].get(risk_level, COUNSELING_KNOWLEDGE[intent]["responses"]["Low"])
        # Add empathy prefix if very negative
        if is_very_negative:
            response = "I can sense you're going through a lot right now, and I want you to know I'm here to support you. " + response
        return response
    
    # Fallback response system
    fallbacks = [
        f"I'm here to help, {username}. Could you tell me more about your academic goals or anything that's stressing you out?",
        "That's an interesting point. Many students find that discussing their routine helps clarify their focus. Would you like to build a study plan together?",
        "I'm specialized in academic success and stress management. Feel free to ask about your CGPA, attendance, or how to handle a heavy workload.",
        "I'm listening. Sometimes just putting your thoughts into words can reduce anxiety. What else is on your mind?"
    ]
    return random.choice(fallbacks)
