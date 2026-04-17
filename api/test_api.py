import requests
import json
import time

def test_api():
    print("Testing Student Dropout Prediction API...")
    url = "http://127.0.0.1:5000/predict"
    
    # 1. Test standard expected data
    good_payload = {
        "gender": "Female",
        "age": 20,
        "year_of_study": 2,
        "department": "Computer Science",
        "attendance_percentage": 95,
        "cgpa": 8.5,
        "backlogs": 0,
        "internal_marks_avg": 8.0,
        "assignment_submission_rate": 0.9,
        "class_participation_score": 8.5,
        "login_frequency_lms": "High",
        "late_submission_count": 0,
        "disciplinary_warnings": 0,
        "extracurricular_participation": "Yes",
        "library_usage": "High",
        "doubt_forum_activity": "High",
        "ai_tool_usage": 1,
        "sleep_hours": 7.5,
        "family_income_range": "High",
        "parental_education": "Postgraduate",
        "part_time_job": "No",
        "commute_time_minutes": 30,
        "internet_access": "High",
        "online_class_attendance": 0.9,
        "recorded_lecture_views": 15,
        "exam_anxiety": "Low",
        "self_confidence_score": 8.5
    }
    
    print("\n--- Test 1: Good Payload (Expected Low Risk) ---")
    try:
        response = requests.post(url, json=good_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        
    # 2. Test missing fields (API should impute to 0/NaN appropriately)
    partial_payload = {
        "attendance_percentage": 35,
        "cgpa": 5.0,
        "backlogs": 4
    }
    
    print("\n--- Test 2: Partial Payload (Expected High Risk) ---")
    try:
        response = requests.post(url, json=partial_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        
    # 3. Test Invalid / Empty 
    print("\n--- Test 3: Empty Payload (Expected Error 400) ---")
    try:
        response = requests.post(url, json={})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        
if __name__ == "__main__":
    test_api()
