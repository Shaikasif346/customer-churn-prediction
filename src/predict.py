"""
Prediction Utilities
Load trained model and make predictions on new customer data
"""

import pickle
import numpy as np
import pandas as pd
import os


def load_model(model_path='models/best_model.pkl'):
    """Load the saved trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Model loaded: {data['name']}")
    return data['model'], data['scaler'], data['features']


def predict_single(customer_data: dict, model_path='models/best_model.pkl'):
    """
    Predict churn for a single customer.

    Args:
        customer_data (dict): Customer feature dictionary
        model_path (str): Path to saved model

    Returns:
        dict: prediction, probability, confidence
    """
    model, scaler, features = load_model(model_path)

    df = pd.DataFrame([customer_data])

    # Feature engineering (must match preprocess.py)
    df['charges_per_month_ratio'] = df['total_charges'] / (df['tenure'] + 1)
    df['is_new_customer'] = (df['tenure'] < 12).astype(int)
    df['has_support'] = ((df['tech_support'] == 1) | (df['online_security'] == 1)).astype(int)
    df['num_services'] = (df[['phone_service', 'multiple_lines',
                               'online_security', 'tech_support', 'streaming_tv']].sum(axis=1))

    # Align columns to training features
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    confidence = "High" if probability > 0.75 or probability < 0.25 else "Medium"
    if 0.45 <= probability <= 0.55:
        confidence = "Low"

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "probability": round(float(probability), 4),
        "confidence": confidence,
        "risk_level": "🔴 High Risk" if probability > 0.7 else
                      "🟡 Medium Risk" if probability > 0.4 else "🟢 Low Risk"
    }


def predict_batch(csv_path: str, model_path='models/best_model.pkl', output_path='outputs/predictions.csv'):
    """Predict churn for a batch of customers from CSV"""
    model, scaler, features = load_model(model_path)

    df = pd.read_csv(csv_path)
    original_df = df.copy()

    if 'customer_id' in df.columns:
        ids = df['customer_id']
        df.drop('customer_id', axis=1, inplace=True)
    if 'churn' in df.columns:
        df.drop('churn', axis=1, inplace=True)

    # Feature engineering
    df['charges_per_month_ratio'] = df['total_charges'] / (df['tenure'] + 1)
    df['is_new_customer'] = (df['tenure'] < 12).astype(int)
    df['has_support'] = ((df['tech_support'] == 1) | (df['online_security'] == 1)).astype(int)
    df['num_services'] = (df[['phone_service', 'multiple_lines',
                               'online_security', 'tech_support', 'streaming_tv']].sum(axis=1))

    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    result_df = original_df.copy()
    result_df['predicted_churn'] = predictions
    result_df['churn_probability'] = np.round(probabilities, 4)
    result_df['risk_level'] = pd.cut(probabilities,
                                      bins=[0, 0.4, 0.7, 1.0],
                                      labels=['Low Risk', 'Medium Risk', 'High Risk'])

    os.makedirs('outputs', exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Batch predictions saved: {output_path}")
    print(f"   Total customers  : {len(result_df)}")
    print(f"   Predicted churns : {predictions.sum()} ({predictions.mean():.1%})")
    return result_df


if __name__ == "__main__":
    # Example single prediction
    sample_customer = {
        'age': 35, 'gender': 1, 'tenure': 6,
        'phone_service': 1, 'multiple_lines': 0,
        'internet_service': 1, 'online_security': 0,
        'tech_support': 0, 'streaming_tv': 1,
        'contract': 0, 'paperless_billing': 1,
        'payment_method': 2,
        'monthly_charges': 75.5, 'total_charges': 453.0
    }

    result = predict_single(sample_customer)
    print("\n🔍 Single Prediction Result:")
    for k, v in result.items():
        print(f"   {k}: {v}")
