import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

def test_drop():
    df = pd.read_excel(r'd:\onedrive old\student_dropout_dataset_4000_rows.xlsx')
    
    # Let's drop the top 2 leaking features
    df = df.drop(['student_id', 'motivation_level', 'stress_level'], axis=1, errors='ignore')
    
    X = df.drop('dropout_risk', axis=1)
    y = df['dropout_risk']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
    
    if len(categorical_features) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[numeric_features]), columns=numeric_features)
    for col in categorical_features:
        X_scaled[col] = X[col]
        
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    print("Accuracy after dropping motivation_level, stress_level:", model.score(X_test, y_test))

if __name__ == "__main__":
    test_drop()
