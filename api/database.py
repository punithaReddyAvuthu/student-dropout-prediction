import sqlite3
import os
from datetime import datetime

# Path to the database file
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DB_DIR, 'database', 'predictions.db')

def get_db_connection():
    """Establish a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create the predictions table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            -- Student Information
            gender TEXT,
            age INTEGER,
            year_of_study INTEGER,
            department TEXT,
            
            -- Academic Performance
            attendance_percentage REAL,
            cgpa REAL,
            backlogs INTEGER,
            internal_marks_avg REAL,
            
            -- Engagement & Behavior
            assignment_submission_rate REAL,
            class_participation_score REAL,
            login_frequency_lms TEXT,
            late_submission_count INTEGER,
            disciplinary_warnings INTEGER,
            extracurricular_participation TEXT,
            library_usage TEXT,
            doubt_forum_activity TEXT,
            ai_tool_usage INTEGER,
            
            -- Personal & Environmental Factors
            sleep_hours REAL,
            family_income_range TEXT,
            parental_education TEXT,
            part_time_job TEXT,
            commute_time_minutes INTEGER,
            internet_access TEXT,
            
            -- Online Learning Specifics
            online_class_attendance REAL,
            recorded_lecture_views INTEGER,
            exam_anxiety TEXT,
            self_confidence_score REAL,
            
            -- Prediction Output
            prediction_probability REAL,
            predicted_risk_level TEXT,
            xai_rationale_shap TEXT,
            xai_rationale_lime TEXT,
            recommendations TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_FILE}")

def log_prediction(input_data, probability, risk_level, xai_shap=None, xai_lime=None, recommendations=None):
    """Log a prediction to the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Prepare the data dictionary with default None for missing keys
        data = {k: input_data.get(k, None) for k in [
            'gender', 'age', 'year_of_study', 'department', 
            'attendance_percentage', 'cgpa', 'backlogs', 'internal_marks_avg',
            'assignment_submission_rate', 'class_participation_score', 
            'login_frequency_lms', 'late_submission_count', 'disciplinary_warnings',
            'extracurricular_participation', 'library_usage', 'doubt_forum_activity', 
            'ai_tool_usage', 'sleep_hours', 'family_income_range', 'parental_education', 
            'part_time_job', 'commute_time_minutes', 'internet_access', 
            'online_class_attendance', 'recorded_lecture_views', 'exam_anxiety', 
            'self_confidence_score'
        ]}
        
        # Add timestamp and prediction outputs
        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data['prediction_probability'] = float(probability)
        data['predicted_risk_level'] = risk_level
        data['xai_rationale_shap'] = xai_shap
        data['xai_rationale_lime'] = xai_lime
        data['recommendations'] = recommendations
        
        # Build query dynamically based on keys
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        values = tuple(data.values())
        
        cursor.execute(f'''
            INSERT INTO predictions ({columns})
            VALUES ({placeholders})
        ''', values)
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        print(f"Error logging to database: {str(e)}")
        return False

# Initialize the DB when this file is imported or run
if __name__ == '__main__':
    init_db()
else:
    init_db()
