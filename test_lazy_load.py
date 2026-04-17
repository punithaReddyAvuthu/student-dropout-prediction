import requests
import json
import time

API_URL = "http://localhost:5000"

def test_chat():
    print("\n💬 Testing Chat (Expect Lazy Load)...")
    start = time.time()
    resp = requests.post(f"{API_URL}/chat", json={"query": "I feel stressed about my exams"})
    duration = time.time() - start
    print(f"Chat Response: {resp.json().get('response')}")
    print(f"Latency: {duration:.2f}s")

def test_predict():
    print("\n🔮 Testing Prediction (Expect Lazy Load XAI)...")
    payload = {
        "gender": "Male", "age": 20, "year_of_study": 2, "department": "CSE",
        "attendance_percentage": 40, "cgpa": 2.0, "backlogs": 5,
        "internal_marks_avg": 30, "class_participation_score": 2,
        "login_frequency_lms": 1, "late_submission_count": 10,
        "disciplinary_warnings": 1, "stress_level": 9,
        "exam_anxiety": 9, "sleep_hours": 4, "motivation_level": 2, 
        "self_confidence_score": 2, "assignment_submission_rate": 40, 
        "extracurricular_participation": 0, "library_usage": 1, 
        "doubt_forum_activity": 1, "ai_tool_usage": 1, "family_income_range": "Low", 
        "parental_education": "School", "part_time_job": 1, "commute_time_minutes": 60, 
        "internet_access": 0, "online_class_attendance": 20, "recorded_lecture_views": 2
    }
    start = time.time()
    resp = requests.post(f"{API_URL}/predict", json=payload)
    duration = time.time() - start
    data = resp.json()
    print(f"Risk: {data['prediction']['risk_level']}")
    print(f"XAI Factors: {data['prediction']['why_this_risk'][:2]}")
    print(f"Latency: {duration:.2f}s")

if __name__ == "__main__":
    # Give the API a moment to breathe
    time.sleep(2)
    test_chat()
    test_predict()
