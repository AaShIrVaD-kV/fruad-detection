import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def load_data_from_csv(file_path="fraud_transactions.csv"):
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file '{file_path}' loaded successfully!")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Assuming the last column is the target variable
    target_column = df.columns[-1]
    
    # Fill missing values with column mean
    df.fillna(df.mean(), inplace=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

# Train model
def train_model():
    data = load_data_from_csv()
    
    if data is None:
        print("No data loaded. Exiting...")
        return
    
    X, y = data
    
    # Stratified split to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Balancing data using SMOTE...")
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Gradient Boosting model (this may take a moment)...")
    # GradientBoostingClassifier is generally more accurate than Random Forest
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train_bal)
    
    y_pred = model.predict(X_test_scaled)
    print("Model Performance:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    joblib.dump(model, "improved_fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_model()
# Output:
# CSV file 'C:/Users/username/Desktop/creditcard.csv' loaded successfully!  # User selects a CSV file from a file dialog
# Model Performance: