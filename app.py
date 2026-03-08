"""
Flask REST API for Customer Churn Prediction
Provides real-time churn prediction endpoint
"""

from flask import Flask, request, jsonify, render_template_string
import traceback
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_single, predict_batch

app = Flask(__name__)

# ── HTML UI ──────────────────────────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #333; }
        header { background: linear-gradient(135deg, #1A3E6E, #2E75B6); color: white;
                 padding: 24px 40px; }
        header h1 { font-size: 1.8rem; }
        header p { opacity: 0.85; margin-top: 4px; }
        .container { max-width: 700px; margin: 40px auto; padding: 0 20px; }
        .card { background: white; border-radius: 12px; padding: 32px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 24px; }
        .card h2 { color: #1A3E6E; margin-bottom: 20px; font-size: 1.2rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        label { font-size: 0.85rem; color: #555; font-weight: 600; display: block; margin-bottom: 5px; }
        input, select { width: 100%; padding: 10px 12px; border: 1.5px solid #ddd;
                        border-radius: 8px; font-size: 0.95rem; transition: border 0.2s; }
        input:focus, select:focus { outline: none; border-color: #2E75B6; }
        button { width: 100%; padding: 14px; background: linear-gradient(135deg, #1A3E6E, #2E75B6);
                 color: white; border: none; border-radius: 8px; font-size: 1rem;
                 font-weight: 600; cursor: pointer; margin-top: 20px; transition: opacity 0.2s; }
        button:hover { opacity: 0.9; }
        #result { display: none; }
        .result-box { border-radius: 10px; padding: 20px; text-align: center; }
        .churn { background: #fff5f5; border: 2px solid #F44336; }
        .no-churn { background: #f0fff4; border: 2px solid #4CAF50; }
        .result-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 8px; }
        .churn .result-title { color: #c62828; }
        .no-churn .result-title { color: #2e7d32; }
        .prob { font-size: 1.1rem; margin: 6px 0; }
        .tag { display: inline-block; padding: 4px 14px; border-radius: 20px;
               font-size: 0.85rem; font-weight: 600; margin-top: 8px; }
        .tag-high { background: #ffebee; color: #c62828; }
        .tag-medium { background: #fff8e1; color: #f57f17; }
        .tag-low { background: #e8f5e9; color: #2e7d32; }
    </style>
</head>
<body>
<header>
    <h1>📊 Customer Churn Predictor</h1>
    <p>AI-powered prediction system — Built by Shaik Asif</p>
</header>
<div class="container">
    <div class="card">
        <h2>Enter Customer Details</h2>
        <div class="grid">
            <div><label>Age</label><input type="number" id="age" value="35" min="18" max="100"></div>
            <div><label>Tenure (months)</label><input type="number" id="tenure" value="6" min="1" max="72"></div>
            <div><label>Monthly Charges ($)</label><input type="number" id="monthly" value="75.5" step="0.1"></div>
            <div><label>Total Charges ($)</label><input type="number" id="total" value="453" step="0.1"></div>
            <div><label>Contract Type</label>
                <select id="contract">
                    <option value="0">Month-to-Month</option>
                    <option value="1">One Year</option>
                    <option value="2">Two Year</option>
                </select>
            </div>
            <div><label>Internet Service</label>
                <select id="internet">
                    <option value="0">None</option>
                    <option value="1" selected>DSL</option>
                    <option value="2">Fiber Optic</option>
                </select>
            </div>
            <div><label>Tech Support</label>
                <select id="tech">
                    <option value="0" selected>No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div><label>Online Security</label>
                <select id="security">
                    <option value="0" selected>No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
        </div>
        <button onclick="predict()">🔍 Predict Churn</button>
    </div>
    <div class="card" id="result">
        <h2>Prediction Result</h2>
        <div id="result-box" class="result-box">
            <div class="result-title" id="result-title"></div>
            <div class="prob" id="result-prob"></div>
            <div id="result-risk"></div>
        </div>
    </div>
</div>
<script>
async function predict() {
    const data = {
        age: +document.getElementById('age').value,
        gender: 1, tenure: +document.getElementById('tenure').value,
        phone_service: 1, multiple_lines: 0,
        internet_service: +document.getElementById('internet').value,
        online_security: +document.getElementById('security').value,
        tech_support: +document.getElementById('tech').value,
        streaming_tv: 0,
        contract: +document.getElementById('contract').value,
        paperless_billing: 1, payment_method: 2,
        monthly_charges: +document.getElementById('monthly').value,
        total_charges: +document.getElementById('total').value
    };
    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    const result = await res.json();
    const box = document.getElementById('result-box');
    box.className = 'result-box ' + (result.prediction === 'Churn' ? 'churn' : 'no-churn');
    document.getElementById('result-title').textContent = result.prediction === 'Churn' ? '⚠️ Customer Will Churn' : '✅ Customer Will Stay';
    document.getElementById('result-prob').textContent = `Churn Probability: ${(result.probability * 100).toFixed(1)}%`;
    document.getElementById('result-risk').innerHTML = `<span class="tag tag-${result.confidence.toLowerCase()}">${result.risk_level}</span>`;
    document.getElementById('result').style.display = 'block';
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        result = predict_single(data)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({'error': 'Model not found. Please run: python src/train.py first'}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'Customer Churn Predictor v1.0'})


if __name__ == '__main__':
    print("🚀 Starting Customer Churn Prediction API...")
    print("   URL: http://localhost:5000")
    print("   API: POST /predict")
    app.run(debug=True, host='0.0.0.0', port=5000)
