from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import sqlite3
import random
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import traceback
import shap
import lime
import lime.lime_tabular

# Import our database script to log predictions and manage users
from database import log_prediction, add_user, verify_user, get_db_connection
from chatbot_engine import get_chatbot_response

app = Flask(__name__)
CORS(app) # Enable CORS for all routes to allow the Streamlit dashboard to connect

# --- 1. Load the Model, Encoder, and XAI Background ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_catboost.pkl')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
BG_PATH = os.path.join(MODEL_DIR, 'xai_background.pkl')

# Global variables for model and encoders
model = None
le_target = None
X_background = None # Keep X_background as global
shap_explainer = None
lime_explainer = None
scaler = None
feature_encoders = None

# Load model and encoders at startup
def load_all_artifacts():
    global model, le_target, X_background, shap_explainer, lime_explainer, scaler, feature_encoders
    try:
        # Standardize paths using os.path.dirname(__file__)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'models')
        
        model_path = os.path.join(model_dir, 'best_model_catboost.pkl')
        le_path = os.path.join(model_dir, 'label_encoder.pkl')
        bg_path = os.path.join(model_dir, 'xai_background.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        enc_path = os.path.join(model_dir, 'feature_encoders.pkl')
        
        model = joblib.load(model_path)
        le_target = joblib.load(le_path)
        X_background = joblib.load(bg_path) # Load X_background here
        
        # Load preprocessing artifacts
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        if os.path.exists(enc_path):
            feature_encoders = joblib.load(enc_path)

        print("Successfully loaded model and artifacts.")
        
    except Exception as e:
        print(f"Error loading model/artifacts: {e}")

def get_shap_explainer():
    """Lazy load SHAP explainer."""
    global shap_explainer
    if shap_explainer is None and model is not None:
        try:
            print("... Initializing SHAP TreeExplainer (optimized for Trees) ...")
            # For tree-based models like CatBoost/XGBoost, 
            # we don't strictly need background data for TreeExplainer, 
            # and it's much faster/stabler without it.
            shap_explainer = shap.TreeExplainer(model)
            print("[SUCCESS] SHAP initialized.")
        except Exception as e:
            print(f"SHAP Init Error: {e}")
    return shap_explainer

def get_lime_explainer():
    """Lazy load LIME explainer."""
    global lime_explainer
    if lime_explainer is None and X_background is not None and le_target is not None: # Added le_target check
        try:
            print("... Initializing LIME TabularExplainer ...")
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_background.values,
                feature_names=X_background.columns.tolist(),
                class_names=le_target.classes_.tolist(),
                mode='classification'
            )
            print("[SUCCESS] LIME initialized.")
        except Exception as e:
            print(f"LIME Init Error: {e}")
    return lime_explainer

# Load immediately on startup
load_all_artifacts()

# Final feature lists for consistency (21 numeric, 8 categorical = 29 total)
NUMERIC_FEATURES = [
    'age', 'year_of_study', 'attendance_percentage', 'cgpa', 'backlogs', 
    'internal_marks_avg', 'assignment_submission_rate', 'class_participation_score', 
    'login_frequency_lms', 'late_submission_count', 'disciplinary_warnings', 
    'extracurricular_participation', 'self_confidence_score', 'stress_level', 
    'motivation_level', 'sleep_hours', 'part_time_job', 'commute_time_minutes', 
    'online_class_attendance', 'recorded_lecture_views', 'ai_tool_usage'
]

CATEGORICAL_FEATURES = [
    'gender', 'department', 'family_income_range', 'parental_education', 
    'library_usage', 'exam_anxiety', 'internet_access', 'doubt_forum_activity'
]

TRAINING_FEATURES = [
    'gender', 'age', 'year_of_study', 'department', 'attendance_percentage', 
    'cgpa', 'backlogs', 'internal_marks_avg', 'assignment_submission_rate', 
    'class_participation_score', 'login_frequency_lms', 'late_submission_count', 
    'disciplinary_warnings', 'extracurricular_participation', 'library_usage', 
    'self_confidence_score', 'stress_level', 'motivation_level', 'exam_anxiety', 
    'sleep_hours', 'family_income_range', 'parental_education', 'part_time_job', 
    'commute_time_minutes', 'internet_access', 'online_class_attendance', 
    'recorded_lecture_views', 'doubt_forum_activity', 'ai_tool_usage'
]

def get_recommendations(risk_level, shap_results):
    """
    Generate high-impact, professional recommendations categorized by persona.
    """
    if not shap_results:
        return {
            "teacher": "Maintain current curriculum engagement. Monitor next internal assessment.",
            "counselor": "Student appears stable. Routine check-in recommended in 30 days.",
            "summary": "Excellent stability observed across all parameters."
        }

    strengths = []
    risks = []
    
    for result in shap_results:
        if isinstance(result, dict):
            feature = result['feature']
            if result['status'] == 'Good': strengths.append(feature)
            else: risks.append(feature)
        else:
            # Fallback for old string format during transition
            part = result.split(":")
            if len(part) < 2: continue
            if "strengthens" in str(result) or "consistent" in str(result): strengths.append(part[0].strip())
            else: risks.append(part[0].strip())

    if risk_level == "High":
        summary = "CRITICAL INTERVENTION NEEDED: The student is showing significant signs of academic detachment."
        teacher = f"URGENT: Focus on recovering {', '.join(risks[:2])}. Consider an alternate assignment format to re-engage the student."
        counselor = "HIGH PRIORITY: Conduct a one-on-one session. Explore external stress factors and provide immediate mental health support."
    
    elif risk_level == "Medium":
        summary = "PREVENTATIVE CARE RECOMMENDED: Stable but showing early warning signs in specific areas."
        teacher = f"Provide additional resources for {', '.join(risks[:2])}. Peer-mentoring in these subjects could be highly effective."
        counselor = "Early intervention advised. Student might be feeling overwhelmed. Recommend a time-management workshop."
    
    else: # Low Risk
        summary = "EXCELLENT STANDING: Student is performing at a high level with strong engagement."
        teacher = f"Encourage leadership roles. Their strength in {', '.join(strengths[:2])} can be used to help peers."
        counselor = "Continue periodic encouragement. They are a role model for consistency."

    return {
        "summary": summary,
        "teacher": teacher,
        "counselor": counselor,
        "checklist": [f"Improve {r}" for r in risks] + [f"Maintain {s}" for s in strengths]
    }

# We need to simulate the exact preprocessing steps from train_models.py
def preprocess_input(input_dict):
    """
    Apply the exact preprocessing strategy used during training.
    """
    # Create DataFrame from input dictionary (single row)
    df = pd.DataFrame([input_dict])
    
    # Ensure all required columns exist, fill missing with NaN
    X = pd.DataFrame(index=[0], columns=TRAINING_FEATURES)
    for col in TRAINING_FEATURES:
        if col in df.columns:
            X[col] = df[col].iloc[0]
        else:
            X[col] = np.nan
            
    # 1. Handle Categorical Features (Match training logic: cast to str)
    for col in CATEGORICAL_FEATURES:
        # Fill missing with 'Unknown'
        if pd.isna(X[col].iloc[0]):
            X[col] = 'Unknown'
        
        # Ensure it is a string for the LabelEncoder
        val = str(X[col].iloc[0])
        
        if feature_encoders and col in feature_encoders:
            le = feature_encoders[col]
            if val in le.classes_:
                X[col] = le.transform([val])
            else:
                # If unseen, use the first class from the encoder as a fallback
                X[col] = le.transform([le.classes_[0]])
        else:
            X[col] = 0

    # 2. Handle Numerical Features (Match training logic)
    # Fill missing numerics with 0
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(0)
    
    # Scale numeric features using SAVED scaler
    if scaler:
        # Scale ONLY the 21 numeric features.
        X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES].astype(float).values)
    
    return X

# --- 3. Define the /predict Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not le_target:
        return jsonify({"error": "Model or Label Encoder not loaded."}), 500
        
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No JSON payload provided."}), 400
            
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    try:
        # 3b. Preprocess the Data
        X_processed = preprocess_input(input_data)
        
        # --- 3b. Make Prediction ---
        # risk_map (alphabetical): High=0, Low=1, Medium=2
        risk_map = {0: "High", 1: "Low", 2: "Medium"}
        prediction = model.predict(X_processed)
        pred_class = int(prediction[0])
        risk_level = risk_map.get(pred_class, "Unknown")
        
        # 3c. Get Prediction Score (Probability)
        probabilities = model.predict_proba(X_processed)[0]
        prediction_score = float(probabilities[pred_class])
        
        # --- 3d. Generate XAI Explanations ---
        explanations = []
        
        # 3d.1 SHAP Explanation
        explainer = get_shap_explainer()
        if explainer:
            try:
                # Optimized SHAP logic
                shap_output = explainer.shap_values(X_processed)
                if isinstance(shap_output, list):
                    row_shap = shap_output[pred_class][0]
                elif len(shap_output.shape) == 3:
                    row_shap = shap_output[0, :, pred_class]
                else:
                    row_shap = shap_output[0]
                
                # ... same logic as before for labels ...
                # --- Refined Factor Analysis ---
                top_indices = np.argsort(np.abs(row_shap))[-4:][::-1]
                for idx in top_indices:
                    feature_name = list(X_processed.columns)[idx].replace('_', ' ').title()
                    val = row_shap[idx]
                    
                    # Logic for "Good/Bad" categorization
                    is_good = (val < 0) # Negative SHAP value reduces dropout risk
                    
                    status = "Good" if is_good else "Danger"
                    if abs(val) < 0.05: status = "Warning" # Subtle impact
                    
                    # Vocabulary Upgrade
                    if is_good:
                        label = random.choice(["Top Performance Lever", "Strong Foundation", "Academic Asset", "Strength Field"])
                        advice = f"Your {feature_name} is currently a huge win. Keep it up!"
                    else:
                        label = random.choice(["Urgent Priority", "Danger Zone", "Growth Opportunity", "Critical Barrier"])
                        advice = f"Your {feature_name} is pulling your performance down. Let's fix this."

                    explanations.append({
                        "feature": feature_name,
                        "status": status,
                        "label": label,
                        "advice": advice
                    })
            except Exception as e:
                print(f"SHAP Runtime Error: {e}")
                explainer = None
        
        if not explanations:
            # Simple fallback
            explanations = [{"feature": "General Engagement", "status": "Good", "label": "Stable Data", "advice": "Data profile appears consistent."}]
        
        # 3d.2 LIME Explanation
        lime_rationale = ""
        l_explainer = get_lime_explainer()
        if l_explainer:
            # LIME explainer needs the prediction function
            exp = l_explainer.explain_instance(
                X_processed.values[0], 
                model.predict_proba, 
                num_features=3,
                labels=[pred_class]
            )
            # Get list of (feature, weight)
            lime_list = exp.as_list(label=pred_class)
            lime_rationale = ", ".join([f"{f}" for f, w in lime_list])

        # --- 3e. Generate Recommendations ---
        recommendations = get_recommendations(risk_level, explanations)

        # 3f. Logging to SQLite
        # Convert structured explanations to strings for database storage
        shap_logs = [f"{e['feature']} ({e['status']})" for e in explanations]
        shap_str = " | ".join(shap_logs)
        
        log_prediction(input_data, prediction_score, risk_level, 
                      xai_shap=shap_str, xai_lime=lime_rationale,
                      recommendations=str(recommendations))
        
        # 3g. Return Response
        return jsonify({
            "status": "success",
            "prediction": {
                "risk_level": risk_level,
                "confidence_score": round(prediction_score, 4),
                "why_this_risk": explanations,
                "lime_rationale": lime_rationale,
                "recommendations": recommendations
            },
            "message": "Student evaluated successfully with XAI and Recommendations."
        }), 200

    except Exception as e:
        print(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to process prediction.",
            "details": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        risk_level = data.get('risk_level', 'Low')
        username = data.get('username', 'Student')
        
        if not user_query:
            return jsonify({"error": "No query provided."}), 400
        
        # Get AI response with context
        bot_response = get_chatbot_response(user_query, risk_level=risk_level, username=username)
        
        return jsonify({
            "status": "success",
            "response": bot_response
        }), 200
        
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": "Failed to generate chat response."}), 500

# --- 4. Authentication Endpoints ---

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password are required."}), 400
            
        success, message = add_user(username, password, email=email)
        if success:
            return jsonify({"status": "success", "message": message}), 201
        else:
            return jsonify({"error": message}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password are required."}), 400
            
        success, message = verify_user(username, password)
        if success:
            return jsonify({"status": "success", "message": message, "username": username}), 200
        else:
            return jsonify({"error": message}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500
            
@app.route('/api/data', methods=['GET'])
def get_prediction_data():
    """Endpoint for the dashboard to fetch all historical records."""
    try:
        conn = get_db_connection()
        # Fetching column names to convert to dictionary
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        data = [dict(row) for row in rows]
        conn.close()
        
        return jsonify({"status": "success", "data": data}), 200
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return jsonify({"error": "Failed to fetch student records"}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online", 
        "model_loaded": model is not None,
        "api": "Student Dropout Prediction API"
    }), 200

if __name__ == '__main__':
    # Initialize port, debug mode
    print("Starting Flask API for Student Dropout Prediction...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
