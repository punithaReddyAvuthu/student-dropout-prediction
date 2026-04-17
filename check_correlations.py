import pandas as pd
from sklearn.preprocessing import LabelEncoder

def analyze_leakage():
    try:
        df = pd.read_excel(r'd:\onedrive old\student_dropout_dataset_4000_rows.xlsx')
        print("Data loaded successfuly.")
        
        # Identify non-numeric columns and encode them temporarily just to get correlations
        non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in non_numeric_cols:
            le = LabelEncoder()
            df[col] = pd.Series(le.fit_transform(df[col][df[col].notnull()]), index=df[col][df[col].notnull()].index)

        # Drop student ID if present
        if 'student_id' in df.columns:
            df = df.drop('student_id', axis=1)

        correlations = df.corr()['dropout_risk'].sort_values(ascending=False)
        print("\n--- Correlations with dropout_risk ---")
        print(correlations)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    analyze_leakage()
