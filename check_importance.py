import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

def find_leakage():
    print("Loading data...")
    df = pd.read_excel(r'd:\onedrive old\student_dropout_dataset_4000_rows.xlsx')
    
    if 'student_id' in df.columns:
        df = df.drop('student_id', axis=1)
        
    X = df.drop('dropout_risk', axis=1)
    y = df['dropout_risk']
    
    # Identify numerical and categorical features
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
    # keeping the dataframe so we get column names back
    X_scaled = pd.DataFrame(scaler.fit_transform(X[numeric_features]), columns=numeric_features)
    for col in categorical_features:
        X_scaled[col] = X[col]
        
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    print("Test Accuracy:", model.score(X_test, y_test))
    
    # Feature importances
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': X_scaled.columns, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
    print("\n--- Feature Importances ---")
    print(feature_imp.head(10))

if __name__ == "__main__":
    find_leakage()
