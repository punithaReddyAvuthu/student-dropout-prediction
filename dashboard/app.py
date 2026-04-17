import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Dropout Prediction Dashboard",
    page_icon="Student",
    layout="wide"
)

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
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api', 'database', 'predictions.db')

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
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

    # --- API Status Check ---
    try:
        import requests
        resp = requests.get("http://localhost:5000/", timeout=5)
        api_status = "Online" if resp.status_code == 200 else "Offline"
    except:
        api_status = "Offline"
    
    if api_status == "Online":
        st.sidebar.success("Backend API: ONLINE")
    else:
        st.sidebar.error("Backend API: OFFLINE")
        if st.sidebar.button("🔄 Start Backend Server"):
            with st.spinner("Starting API..."):
                import subprocess
                import sys
                import os
                api_path = os.path.join(os.getcwd(), 'api', 'app.py')
                subprocess.Popen([sys.executable, api_path], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
                st.rerun()

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
            try:
                import requests
                response = requests.post("http://localhost:5000/predict", json=payload)
                if response.status_code == 200:
                    res = response.json()['prediction']
                    st.success(f"Risk Level: **{res['risk_level']}** (Confidence: {res['confidence_score']})")
                    st.info(res['recommendations'])
                    st.write("#### Contributing Factors:")
                    for f in res['why_this_risk']: st.write(f"- {f}")
                    st.balloons()
            except Exception as e: 
                st.error(f"📡 Connection Error: Could not reach the Backend API.")
                st.info("💡 Try clicking the 'Start Backend Server' button in the sidebar or make sure 'python start_system.py' is running in your terminal.")
                st.debug(f"Details: {e}")

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

            # Get response from API
            try:
                # Use a small delay for "AI thinking" feel
                with st.spinner("AI Counselor is thinking..."):
                    resp = requests.post("http://localhost:5000/chat", json={"query": user_input}, timeout=30)
                    if resp.status_code == 200:
                        bot_res = resp.json().get("response", "I'm listening. Please continue.")
                    else:
                        bot_res = "I'm having a little trouble connecting right now, but I'm here for you. Is there anything else on your mind?"
            except Exception:
                bot_res = "Connection error. Please ensure the backend server is running."

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
