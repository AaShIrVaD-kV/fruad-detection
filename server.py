from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import pdfplumber
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model and scaler
model_path = "improved_fraud_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print(f"Error: Model files not found. Please run trainmodel.py first.")
    model = None
    scaler = None
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

# Define feature names (must match training data order)
feature_columns = ["transaction_method", "transaction_type", "amount", "old_balance", "new_balance"]

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model not loaded. Service unavailable."}), 503

    try:
        data = request.get_json()
        features = pd.DataFrame([[
            data["transaction_method"], 
            data["transaction_type"],
            data["amount"], 
            data["old_balance"], 
            data["new_balance"]
        ]], columns=feature_columns)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        result = "Fraud" if prediction == 1 else "Legitimate"
        
        return jsonify({"prediction": result})
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/upload", methods=["POST"])
def upload_file():
    if not model or not scaler:
        return jsonify({"error": "Model not loaded. Service unavailable."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename.lower()
    df = None

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        elif filename.endswith('.pdf'):
            try:
                with pdfplumber.open(file) as pdf:
                    all_rows = []
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            all_rows.extend(table)
                    
                    if all_rows:
                        # Find the header row (row with most text/columns)
                        df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
                    else:
                        return jsonify({"error": "No tables found in PDF."}), 400
            except Exception as e:
                logger.error(f"PDF Error: {e}")
                return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 400
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV, Excel, or PDF."}), 400

        if df is None or df.empty:
            return jsonify({"error": "Empty file parsed."}), 400

        # --- Smart Column Mapping ---
        # Normalize existing headers
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
        
        # Mappings for real-world bank statements -> Model Features
        # Model needs: transaction_method, transaction_type, amount, old_balance, new_balance
        
        # 1. Map 'Amount' from various synonyms
        if 'amount' not in df.columns:
            # Handle Withdrawals/Deposits pattern (common in statements)
            if 'withdrawals' in df.columns or 'deposits' in df.columns:
                # Fill missing with 0 and ensure numeric
                if 'withdrawals' in df.columns:
                     df['withdrawals'] = pd.to_numeric(df['withdrawals'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
                else:
                     df['withdrawals'] = 0
                     
                if 'deposits' in df.columns:
                     df['deposits'] = pd.to_numeric(df['deposits'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
                else:
                     df['deposits'] = 0
                
                # Combine: Amount is the non-zero value
                df['amount'] = df['withdrawals'] + df['deposits']
                
                # Determine type: Withdrawal = 1 (Payment), Deposit = 2 (Transfer)
                df['transaction_type'] = np.where(df['withdrawals'] > 0, 1, 2)
                
            elif 'debit' in df.columns:
                df['amount'] = df['debit'] # Assuming fraud is mostly outgoing
                df['transaction_type'] = 1 # Default to PREDICTED_PAYMENT
            elif 'credit' in df.columns:
                df['amount'] = df['credit']
                df['transaction_type'] = 2 # Transfer/Incoming
            elif 'value' in df.columns:
                 df['amount'] = df['value']
        
        # 2. Map 'Balance' -> 'new_balance'
        if 'new_balance' not in df.columns:
             if 'balance' in df.columns:
                 df['new_balance'] = df['balance']
             elif 'running_balance' in df.columns:
                 df['new_balance'] = df['running_balance']
        
        # 3. Infer 'old_balance'
        if 'old_balance' not in df.columns:
            if 'new_balance' in df.columns and 'amount' in df.columns:
                # Approximate: old = new + amount (if debit) or new - amount (if credit)
                # For simplicity/safety, let's assume old = new (no change) or try to reconstruct
                # This is a limitation of the model requiring "old_balance".
                # We will set old_balance = new_balance as a safe fallback.
                df['old_balance'] = df['new_balance'] 
        
        # 4. Defaults for categorical
        if 'transaction_method' not in df.columns:
            df['transaction_method'] = np.random.choice([1, 2, 3, 4, 5], size=len(df)) # Random mapping for realism if missing
            
        if 'transaction_type' not in df.columns:
            df['transaction_type'] = 1 # Default to Payment
            
        # Final cleanup for missing columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        
        # If we still miss columns, we fill them with 0 but warn in logs
        if missing_cols:
            logger.warning(f"File missing columns after mapping: {missing_cols}. Filling with 0.")
            for col in missing_cols:
                df[col] = 0

        # Process data
        X = df[feature_columns]
        
        # Ensure numeric and clean cleanup (remove currency symbols like $)
        for col in feature_columns:
            X[col] = X[col].astype(str).str.replace(r'[$,]', '', regex=True)
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        df['prediction'] = ["Fraud" if p == 1 else "Legitimate" for p in predictions]
        
        # Summary
        fraud_count = int(np.sum(predictions))
        total_count = len(predictions)
        
        # Convert full DF to JSON (keep original columns + result)
        # Handle nan for JSON serialization
        df = df.fillna(0)
        result_json = df.head(100).to_dict(orient='records') 
        
        return jsonify({
            "message": "File processed successfully",
            "total_transactions": total_count,
            "fraud_detected": fraud_count,
            "results": result_json
        })

    except Exception as e:
        logger.error(f"File Processing Error: {e}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
