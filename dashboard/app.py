import streamlit as st
import os
import time
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import base64
from datetime import datetime

# --- TOP LEVEL CRASH PROTECTOR ---
try:
    # --- API Configuration ---
    # In production, set this in Streamlit Secrets (Secrets manager)
    try:
        BACKEND_URL = st.secrets["BACKEND_URL"]
    except Exception:
        # Fallback for local development when secrets.toml doesn't exist
        BACKEND_URL = "http://localhost:5000"

    # --- Page Configuration ---
    st.set_page_config(
        page_title="AI Student Dropout Prediction",
        page_icon="🎓",
        layout="wide"
    )

except Exception as e:
    st.error("🚨 CRITICAL STARTUP ERROR")
    st.write("The application failed to start. Please check your internet connection or backend URL.")
    st.exception(e)
    st.stop()

# --- AI Model Loading (Cached for performance) ---
def verify_user(username, password):
    """Call the Flask API to verify user credentials."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/login", json={"username": username, "password": password})
        if response.status_code == 200:
            return True, "Login successful"
        return False, response.json().get("error", "Invalid credentials")
    except Exception as e:
        return False, f"Connection Error: {e}"

def add_user(username, password, email=None):
    """Call the Flask API to register a new user."""
    try:
        payload = {"username": username, "password": password, "email": email}
        response = requests.post(f"{BACKEND_URL}/api/register", json=payload)
        if response.status_code == 201:
            return True, "Registration successful"
        return False, response.json().get("error", "Registration failed")
    except Exception as e:
        return False, f"Connection Error: {e}"

def get_chatbot_response(query, risk_level="Low", username="Student"):
    """Call the Flask API for chatbot guidance with context."""
    try:
        payload = {"query": query, "risk_level": risk_level, "username": username}
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            return response.json().get("response", "No response from AI.")
        return f"Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {e}"

def run_api_prediction(input_data):
    """Call the Flask API for model prediction and XAI."""
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=input_data)
        if response.status_code == 200:
            # The API returns the "prediction" object directly
            return response.json().get("prediction")
        st.error(f"API Error: {response.text}")
        return None
    except Exception as e:
        st.error(f"Network Error: {e}")
        return None

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Auth Page Background (Dark Mode) */
    .stApp:has(.auth-page-marker) {
        background: radial-gradient(circle at top right, #111827 0%, #000000 100%) !important;
        background-attachment: fixed !important;
    }

    /* Dark Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        color: #ffffff;
        margin-top: 20px;
    }

    .auth-header {
        text-align: center;
        margin-bottom: 25px;
    }

    .auth-header h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        margin-bottom: 5px !important;
        letter-spacing: -0.5px;
    }

    .auth-header p {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 1rem !important;
    }

    /* Customizing Streamlit Widgets in Auth (Dark Mode) */
    .stApp:has(.auth-page-marker) label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    .stApp:has(.auth-page-marker) .stTextInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        color: #ffffff !important;
    }

    .stApp:has(.auth-page-marker) .stTextInput input::placeholder {
        color: rgba(255, 255, 255, 0.3) !important;
    }

    .stApp:has(.auth-page-marker) .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px !important;
        font-size: 1.1rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        height: auto !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stApp:has(.auth-page-marker) .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.4) !important;
    }

    /* Hide the default radio bubble styling */
    .stApp:has(.auth-page-marker) [data-testid="stRadio"] div[role="radiogroup"] {
        background: rgba(255,255,255,0.05);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stApp:has(.auth-page-marker) [data-testid="stRadio"] label {
        color: white !important;
    }

    .auth-footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading (via API) ---
def load_data():
    """Fetch prediction records from the Flask API instead of local SQLite."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/data")
        if response.status_code == 200:
            data = response.json().get("data", [])
            # Convert back to DataFrame for the dashboard charts
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# --- Authentication UI ---
def login_signup_ui():
    # Hidden marker to trigger specific CSS
    st.markdown("<div class='auth-page-marker'></div>", unsafe_allow_html=True)
    
    # Outer spacing
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Main Split-Screen Container
    col_img, col_form = st.columns([1.2, 1], gap="large")
    
    with col_img:
        st.markdown("<br>", unsafe_allow_html=True)
        img_path = os.path.join(os.path.dirname(__file__), "assets", "login_illustration.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.markdown("<h1 style='color: #ffffff; font-size: 4rem; text-align: center;'>🎓</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #ffffff; text-align: center;'>AI Student Portal</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: rgba(255,255,255,0.6); text-align: center;'>Predictive Analytics Dashboard</p>", unsafe_allow_html=True)

    with col_form:
        # Toggle at the very top (Updated colors for dark mode)
        auth_mode = st.radio(
            "Account Mode", ["Sign In", "Create Account"], 
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        if auth_mode == "Sign In":
            st.markdown("""
                <div class='auth-header'>
                    <h1>Login</h1>
                    <p>Enter your details to access the portal</p>
                </div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            
            # Stylistic extras
            c1, c2 = st.columns([1,1])
            with c1: st.checkbox("Remember me", value=True)
            with c2: st.markdown("<p style='text-align: right; margin-top: 5px;'><a href='#' style='color: #3b82f6; text-decoration: none; font-size: 0.8rem; font-weight: 600;'>Forgot Password?</a></p>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", key="btn_login"):
                if username and password:
                    success, message = verify_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Identity Verified. Entering Portal...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("Please provide valid credentials.")
        
        else:
            st.markdown("""
                <div class='auth-header'>
                    <h1>Registration</h1>
                    <p>Create a new student account</p>
                </div>
            """, unsafe_allow_html=True)
            
            new_user = st.text_input("Username", key="signup_user", placeholder="Choose unique username")
            new_email = st.text_input("Email", key="signup_email", placeholder="Enter your email address")
            new_pass = st.text_input("Password", type="password", key="signup_pass", placeholder="Create strong password")
            confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Repeat password")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Complete Registration", key="btn_signup"):
                if new_user and new_pass and new_email:
                    if new_pass == confirm_pass:
                        success, message = add_user(new_user, new_pass, new_email)
                        if success:
                            st.success(f"✅ {message}")
                            st.info("Please sign in to proceed.")
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.warning("Please fill out all registration fields.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-footer'>© 2024 University Academic Network</div>", unsafe_allow_html=True)

# --- Main Dashboard ---
def main():
    # Sidebar Logout and User Info
    st.sidebar.markdown(f"### Welcome, **{st.session_state.username}**! 👋")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
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
                res = run_api_prediction(payload)
                if res:
                    pred = res # The helper already extracted the 'prediction' key
                    risk = pred['risk_level']
                    conf = pred['confidence_score']
                    recs = pred['recommendations']
                    factors = pred['why_this_risk']

                    # --- 1. Executive Summary Banner ---
                    st.markdown("---")
                    col_metric1, col_metric2 = st.columns([1, 1])
                    with col_metric1:
                        st.metric("Risk Assessment", risk, delta=f"{conf*100:.1f}% Confidence", delta_color="inverse" if risk == "High" else "normal")
                    with col_metric2:
                        st.info(f"**Status Summary:** {recs['summary']}")

                    # --- 2. Deep Insights (Good/Bad/Warning) ---
                    st.markdown("### 🔍 Student DNA Insights")
                    cols = st.columns(len(factors) if factors else 1)
                    for i, factor in enumerate(factors):
                        with cols[i % len(cols)]:
                            status_emoji = "✅" if factor['status'] == "Good" else "⚠️" if factor['status'] == "Warning" else "🚩"
                            bgcolor = "rgba(16, 185, 129, 0.1)" if factor['status'] == "Good" else "rgba(245, 158, 11, 0.1)" if factor['status'] == "Warning" else "rgba(239, 68, 68, 0.1)"
                            bordercolor = "#10b981" if factor['status'] == "Good" else "#f59e0b" if factor['status'] == "Warning" else "#ef4444"
                            
                            st.markdown(f"""
                                <div style="background: {bgcolor}; border: 1px solid {bordercolor}; border-radius: 10px; padding: 15px; height: 160px;">
                                    <h4 style="margin:0; font-size: 0.9rem;">{status_emoji} {factor['label']}</h4>
                                    <p style="font-weight: bold; margin: 5px 0;">{factor['feature']}</p>
                                    <p style="font-size: 0.8rem; opacity: 0.8;">{factor['advice']}</p>
                                </div>
                            """, unsafe_allow_html=True)

                    # --- 3. Expert Action Plans ---
                    st.markdown("<br>", unsafe_allow_html=True)
                    tab_t, tab_c, tab_check = st.tabs(["🍎 Teacher's Strategy", "🧠 Counselor's Advice", "✅ My Growth Checklist"])
                    with tab_t:
                        st.markdown(f"**Instructional Priorities:** {recs['teacher']}")
                    with tab_c:
                        st.markdown(f"**Well-being Focus:** {recs['counselor']}")
                    with tab_check:
                        st.write("Mark these off as you improve:")
                        for item in recs['checklist']:
                            st.checkbox(item)

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
            if st.button("🧘 I'm stressed", use_container_width=True): quick_query = "I'm feeling very overwhelmed and stressed about my studies right now."
        with q2:
            if st.button("📚 Improve CGPA", use_container_width=True): quick_query = "My marks are low, how can I improve my bad CGPA and academic performance?"
        with q3:
            if st.button("⏰ Time Management", use_container_width=True): quick_query = "I'm struggling with managing my time and late submissions."
        with q4:
            if st.button("🚩 Low Attendance", use_container_width=True): quick_query = "I have missed many classes and have low attendance issues."

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

            # Get response with context
            try:
                # Latest risk in the system for context
                current_risk = "Low"
                if not df.empty:
                    current_risk = df.iloc[0]['predicted_risk_level']
                
                with st.spinner("AI Counselor is thinking..."):
                    bot_res = get_chatbot_response(
                        user_input, 
                        risk_level=current_risk, 
                        username=st.session_state.username
                    )
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
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
        
    if not st.session_state.authenticated:
        login_signup_ui()
    else:
        main()
