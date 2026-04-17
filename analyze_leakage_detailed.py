import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def analyze():
    print("Loading data...")
    path = r'd:\onedrive old\student_dropout_dataset_4000_rows.xlsx'
    df = pd.read_excel(path)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Target distribution:\n{df['dropout_risk'].value_counts(normalize=True)}")
    
    target = 'dropout_risk'
    features = [c for c in df.columns if c not in [target, 'student_id']]
    
    # Check for direct leakage (1-to-1 mapping)
    leakage_candidates = []
    for col in features:
        # Number of unique target values per unique feature value
        unique_targets_per_val = df.groupby(col)[target].nunique().max()
        if unique_targets_per_val == 1:
            # Check if it's not just a very high cardinality feature
            if df[col].nunique() < len(df) * 0.5:
                leakage_candidates.append(col)
                
    if leakage_candidates:
        print(f"\nPotential Leakage Candidates (Perfect Predictors per value): {leakage_candidates}")
    else:
        print("\nNo obvious 1-to-1 leakage candidates found.")

    # Prepare data for a quick XGBoost run to see feature importance and evaluation
    X = df[features].copy()
    y = df[target]
    
    # Encode categorical
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    print(f"\nPrediction Probability stats:")
    print(pd.Series(y_prob).describe())
    
    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))

if __name__ == "__main__":
    analyze()
