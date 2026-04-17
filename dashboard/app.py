import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import time
import joblib
import numpy as np
import shap
import lime
import lime.lime_tabular
from datetime import datetime

# Import local modules
import sys
# Add parent dir to path to find /api modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'api'))
# We try to import, but if they are missing (e.g. in cloud without local files), we handle it
try:
    from database import log_prediction
    from chatbot_engine import get_chatbot_response
except ImportError:
    # Fallback for cloud deployment where 'api' might be in root or processed differently
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))
    from database import log_prediction
    from chatbot_engine import get_chatbot_response

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# --- AI Model Loading (Cached for performance) ---
@st.cache_resource
def load_ai_models():
    # Correctly locate the project root (one level up from /dashboard)
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    model_dir = os.path.join(project_root, 'api', 'models')
    
    model_file = os.path.join(model_dir, 'best_model_catboost.pkl')
    
    # Check if models exist
    if not os.path.exists(model_file):
        # Fallback: check if we are already in root
        model_dir = os.path.join(app_dir, 'api', 'models')
        model_file = os.path.join(model_dir, 'best_model_catboost.pkl')
        if not os.path.exists(model_file):
            st.error(f"❌ AI Models not found at {model_file}. Please check folder structure.")
            return None

    try:
        artifacts = {
            'model': joblib.load(model_file),
            'le_target': joblib.load(os.path.join(model_dir, 'label_encoder.pkl')),
            'X_background': joblib.load(os.path.join(model_dir, 'xai_background.pkl')),
            'scaler': joblib.load(os.path.join(model_dir, 'scaler.pkl')),
            'feature_encoders': joblib.load(os.path.join(model_dir, 'feature_encoders.pkl'))
        }
        return artifacts
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None

# Global feature lists
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

def preprocess_input(input_dict, artifacts):
    df = pd.DataFrame([input_dict])
    X = pd.DataFrame(index=[0], columns=TRAINING_FEATURES)
    for col in TRAINING_FEATURES:
        X[col] = df[col].iloc[0] if col in df.columns else np.nan
        
    for col in CATEGORICAL_FEATURES:
        if pd.isna(X[col].iloc[0]): X[col] = 'Unknown'
        val = str(X[col].iloc[0])
        le = artifacts['feature_encoders'].get(col)
        if le:
            X[col] = le.transform([val])[0] if val in le.classes_ else le.transform([le.classes_[0]])[0]
        else:
            X[col] = 0

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(0)
    if artifacts['scaler']:
        X[NUMERIC_FEATURES] = artifacts['scaler'].transform(X[NUMERIC_FEATURES].astype(float).values)
    return X

def get_recommendations(risk_level, shap_results):
    if not shap_results: return "Stay focused and maintain your current routine."
    strengths, risks = [], []
    for result in shap_results:
        parts = result.split(":")
        if len(parts) < 2: continue
        if "strengthens" in parts[1] or "helps reduce" in parts[1]: strengths.append(parts[0].strip())
        else: risks.append(parts[0].strip())

    if risk_level == "High":
        return f"URGENT: Schedule counseling. Focus on {', '.join(risks)}. Immediate intervention recommended."
    elif risk_level == "Medium":
        return f"MODERATE: Focus on {', '.join(risks)}. Your strength in {strengths[0] if strengths else 'attendance'} helps."
    return f"LOW RISK: Great job! Consistency in {', '.join(strengths[:2])} is key."

def run_local_prediction(input_data):
    try:
        arts = load_ai_models()
        if not arts: return None
        X_processed = preprocess_input(input_data, arts)
        risk_map = {0: "High", 1: "Low", 2: "Medium"}
        
        prediction = arts['model'].predict(X_processed)
        pred_class = int(prediction[0])
        risk_level = risk_map.get(pred_class, "Unknown")
        
        probs = arts['model'].predict_proba(X_processed)[0]
        score = float(probs[pred_class])
        
        # SHAP
        explainer = shap.TreeExplainer(arts['model'])
        shap_values = explainer.shap_values(X_processed)
        if isinstance(shap_values, list): row_shap = shap_values[pred_class][0]
        elif len(shap_values.shape) == 3: row_shap = shap_values[0, :, pred_class]
        else: row_shap = shap_values[0]
        
        top_idx = np.argsort(np.abs(row_shap))[-3:][::-1]
        explanations = []
        for idx in top_idx:
            feat = list(X_processed.columns)[idx].replace('_', ' ').title()
            explanations.append(f"{feat}: {'increases risk' if risk_level != 'Low' else 'strengthens stability'}")
            
        recommendations = get_recommendations(risk_level, explanations)
        
        # Log to DB
        log_prediction(input_data, score, risk_level, " | ".join(explanations), "", recommendations)
        
        return {
            "risk_level": risk_level,
            "confidence_score": round(score, 4),
            "why_this_risk": explanations,
            "recommendations": recommendations
        }
    except Exception as e:
        import traceback
        st.error(f"Prediction Error: {e}")
        st.code(traceback.format_exc())
        return None

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .kpi-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        border-bottom: 5px solid #ececec;
    }
    .kpi-total { border-bottom-color: #3b82f6; }
    .kpi-high { border-bottom-color: #ef4444; }
    .kpi-low { border-bottom-color: #10b981; }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Connection ---
# Correctly locate the database (one level up from /dashboard, then into /api/database)
def get_db_path():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    return os.path.join(project_root, 'api', 'database', 'predictions.db')

DB_PATH = get_db_path()

def load_data():
    current_db = get_db_path()
    if not os.path.exists(current_db):
        # Fallback to current working directory
        current_db = os.path.join(os.getcwd(), 'api', 'database', 'predictions.db')
        if not os.path.exists(current_db):
            return pd.DataFrame()

    try:
        conn = sqlite3.connect(current_db)
        query = "SELECT * FROM predictions ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Main Dashboard ---
def main():
    st.markdown("<h1 style='text-align: center;'>Student Dropout Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Monitor student risk levels with simple data insights.</p>", unsafe_allow_html=True)
    st.markdown("---")

    df = load_data()

    # --- System Status (Consolidated) ---
    st.sidebar.success("🤖 AI System: Consolidated & Standalone")
    st.sidebar.info("This application handles both prediction and analysis locally (no separate backend needed).")

    if df.empty:
        st.warning("No student data available in the database.")
        if st.button("Refresh"):
            st.rerun()
        return

    # --- KPI Section ---
    total_students = len(df)
    high_risk = len(df[df['predicted_risk_level'] == 'High'])
    medium_risk = len(df[df['predicted_risk_level'] == 'Medium'])
    low_risk = len(df[df['predicted_risk_level'] == 'Low'])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card kpi-total"><div class="kpi-label">Total Students</div><div class="kpi-value">{total_students}</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card kpi-high"><div class="kpi-label">High Risk</div><div class="kpi-value" style="color: #ef4444;">{high_risk}</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card" style="border-bottom: 5px solid #f59e0b;"><div class="kpi-label">Medium Risk</div><div class="kpi-value" style="color: #f59e0b;">{medium_risk}</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card kpi-low"><div class="kpi-label">Low Risk</div><div class="kpi-value" style="color: #10b981;">{low_risk}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts Section ---
    col1, col2 = st.columns(2)
    risk_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}

    with col1:
        st.subheader("Risk Distribution")
        risk_counts = df['predicted_risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig_pie = px.pie(risk_counts, values='Count', names='Risk Level', color='Risk Level', color_discrete_map=risk_colors, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Attendance vs Risk Level")
        avg_attendance = df.groupby('predicted_risk_level')['attendance_percentage'].mean().reset_index()
        fig_bar = px.bar(avg_attendance, x='predicted_risk_level', y='attendance_percentage', color='predicted_risk_level', color_discrete_map=risk_colors)
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Table and Rationale Section ---
    st.markdown("---")
  # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Student Database", 
        "🔍 XAI Explanations (SHAP/LIME)", 
        "🔮 Predict New Student",
        "💬 AI Counseling Chat"
    ])

    with tab1:
        st.subheader("Student Records and Risk Predictions")
        st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)

    with tab2:
        st.subheader("Analyze Individual Student Risk")
        student_ids = df['id'].tolist()
        selected_id = st.selectbox("Select Student ID to explain:", student_ids)
        
        if selected_id:
            student_data = df[df['id'] == selected_id].iloc[0]
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.info(f"Selected Student: ID {selected_id} | Risk Level: {student_data['predicted_risk_level']}")
                st.write("**Top Contributing Factors (SHAP):**")
                shap_rationale = student_data.get('xai_rationale_shap', '')
                if shap_rationale:
                    for item in shap_rationale.split(" | "):
                        st.write(f"- {item}")
                else:
                    st.write("No SHAP data available for this record.")
                    
            with col_right:
                st.write("**Decision Logic (LIME):**")
                lime_rationale = student_data.get('xai_rationale_lime', '')
                if lime_rationale:
                    for item in lime_rationale.split(", "):
                        st.code(item)
                else:
                    st.write("No LIME data available for this record.")

            st.markdown("---")
            st.write("### Personalized Action Plan")
            recommendation_text = student_data.get('recommendations', 'No recommendations available.')
            risk_lvl = student_data['predicted_risk_level']
            if risk_lvl == "High": st.error(recommendation_text)
            elif risk_lvl == "Medium": st.warning(recommendation_text)
            else: st.success(recommendation_text)
            
            st.markdown("---")
            if shap_rationale:
                try:
                    parts = [p.split(": ") for p in shap_rationale.split(" | ")]
                    viz_df = pd.DataFrame(parts, columns=['Feature', 'Impact'])
                    viz_df['Score'] = viz_df['Impact'].apply(lambda x: 1 if 'increases' in x else -1)
                    fig_shap = px.bar(viz_df, x='Score', y='Feature', orientation='h', title="Risk Impact Direction", color='Score', color_continuous_scale=['#10b981', '#ef4444'])
                    fig_shap.update_layout(showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig_shap, use_container_width=True)
                except: pass

    with tab3:
        st.subheader("Interactive Risk Assessment")
        st.markdown("Enter student details below for a full, real-time risk assessment.")
        
        with st.form("prediction_form"):
            with st.expander("Academic & Performance", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5, 0.1)
                    attendance = st.number_input("Attendance %", 0, 100, 85)
                    backlogs = st.number_input("Current Backlogs", 0, 20, 0)
                with col2:
                    int_marks = st.number_input("Internal Marks Avg", 0, 100, 70)
                    part = st.slider("Class Participation (1-10)", 1, 10, 7)
                    sub_rate = st.slider("Assignment Submission Rate (%)", 0, 100, 90)

            with st.expander("Engagement & Behavioral"):
                col1, col2 = st.columns(2)
                with col1:
                    lms_freq = st.number_input("LMS Logins/Week", 0, 7, 5)
                    late_sub = st.number_input("Late Submissions", 0, 50, 1)
                    warn = st.number_input("Disciplinary Warnings", 0, 5, 0)
                    lib_use = st.slider("Library Usage (Hours/Week)", 0, 40, 5)
                    forum = st.slider("Doubt Forum Activity (1-10)", 1, 10, 4)
                with col2:
                    online_att = st.number_input("Online Class Attendance %", 0, 100, 80)
                    lec_views = st.number_input("Recorded Lecture Views", 0, 100, 10)
                    ai_use = st.slider("AI Tool Usage (1-10)", 1, 10, 3)
                    extra = st.selectbox("Extracurricular Participation", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

            with st.expander("Well-being & Personal"):
                col1, col2 = st.columns(2)
                with col1:
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    age = st.number_input("Age", 17, 30, 20)
                    dept = st.selectbox("Department", ["CSE", "ECE", "ME", "CE", "EEE"])
                    year = st.selectbox("Year", [1, 2, 3, 4])
                    sleep = st.number_input("Avg Sleep Hours", 0, 15, 7)
                with col2:
                    stress = st.slider("Stress Level (1-10)", 1, 10, 4)
                    motivation = st.slider("Motivation Level (1-10)", 1, 10, 8)
                    anxiety = st.slider("Exam Anxiety (1-10)", 1, 10, 3)
                    conf = st.slider("Self Confidence (1-10)", 1, 10, 8)

            with st.expander("Socio-Economic & Logistics"):
                col1, col2 = st.columns(2)
                with col1:
                    income = st.selectbox("Family Income Range", ["Low", "Medium", "High"])
                    parent_ed = st.selectbox("Parental Education", ["School", "Bachelor", "Master", "PhD"])
                    internet = st.selectbox("Internet Access", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                with col2:
                    part_time = st.selectbox("Part-time Job", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                    commute = st.number_input("Commute Time (Minutes)", 0, 180, 30)
            
            submit = st.form_submit_button("🌟 Run Real-Time Prediction")
            
        if submit:
            payload = {
                "gender": gender, "age": age, "year_of_study": year, "department": dept,
                "attendance_percentage": attendance, "cgpa": cgpa, "backlogs": backlogs,
                "internal_marks_avg": int_marks, "class_participation_score": part,
                "login_frequency_lms": lms_freq, "late_submission_count": late_sub,
                "disciplinary_warnings": warn, "stress_level": stress,
                "exam_anxiety": anxiety, "sleep_hours": sleep, "motivation_level": motivation, 
                "self_confidence_score": conf, "assignment_submission_rate": sub_rate, 
                "extracurricular_participation": extra, "library_usage": lib_use, 
                "doubt_forum_activity": forum, "ai_tool_usage": ai_use, "family_income_range": income, 
                "parental_education": parent_ed, "part_time_job": part_time, "commute_time_minutes": commute, 
                "internet_access": internet, "online_class_attendance": online_att, "recorded_lecture_views": lec_views
            }
            with st.spinner("🤖 AI Counselor is analyzing your data..."):
                res = run_local_prediction(payload)
                if res:
                    st.success(f"Risk Level: **{res['risk_level']}** (Confidence: {res['confidence_score']})")
                    st.info(res['recommendations'])
                    st.write("#### Contributing Factors:")
                    for f in res['why_this_risk']: st.write(f"- {f}")
                    st.balloons()
                else:
                    st.error("Failed to generate prediction. Check system logs.")

    # --- Tab 4: AI Counseling Chat ---
    with tab4:
        st.markdown("### 💬 Your Personal AI Counseling Assistant")
        st.markdown("*Empathetic guidance for academic success and well-being.*")
        
        # Quick Action Buttons
        st.write("---")
        q1, q2, q3, q4 = st.columns(4)
        quick_query = None
        with q1:
            if st.button("🧘 I'm stressed"): quick_query = "I'm feeling very stressed about my studies right now."
        with q2:
            if st.button("📚 Improve CGPA"): quick_query = "How can I improve my CGPA and academic performance?"
        with q3:
            if st.button("⏰ Time Management"): quick_query = "I'm struggling with managing my time effectively."
        with q4:
            if st.button("🚩 My Risk Factors"): quick_query = "What are my main risk factors for academic struggle?"

        st.write("---")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your AI Counselor. How are you feeling today? Are you stressed about anything or do you need a study plan?"}
            ]

        # Display chat messages from history
        for message in st.session_state.messages:
            avatar = "🎓" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Helper for streaming response
        def response_generator(text):
            for word in text.split():
                yield word + " "
                time.sleep(0.05)

        # Handle user input (either from chat_input or quick buttons)
        user_input = st.chat_input("Ask me anything...")
        if quick_query:
            user_input = quick_query

        if user_input:
            # Add user message to UI
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get response locally
            try:
                # Use a small delay for "AI thinking" feel
                with st.spinner("AI Counselor is thinking..."):
                    bot_res = get_chatbot_response(user_input)
            except Exception as e:
                bot_res = f"I'm having a little trouble thinking right now. Error: {e}"

            # Display assistant response with "Streaming" effect
            with st.chat_message("assistant", avatar="🎓"):
                # st.write_stream is only available in newer streamlit, fallback to simulated stream
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response_generator(bot_res):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": bot_res})
            
            # Rerun to clear any quick_query state if used
            if quick_query: 
                time.sleep(0.5)
                st.rerun()

    if st.button("🔄 Refresh Data"):
        st.rerun()

if __name__ == "__main__":
    main()
