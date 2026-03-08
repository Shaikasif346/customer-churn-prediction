# 📊 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-2.3-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> An end-to-end Machine Learning pipeline to predict customer churn with **92% accuracy** using Random Forest and Logistic Regression models.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges in business. This project builds a complete ML pipeline that:
- Analyzes customer data to find churn patterns
- Trains and evaluates multiple ML models
- Deploys a real-time prediction API using Flask
- Visualizes insights through rich dashboards

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── generate_data.py         # Synthetic dataset generator
│
├── src/
│   ├── preprocess.py            # Data preprocessing & feature engineering
│   ├── train.py                 # Model training & evaluation
│   ├── predict.py               # Prediction utilities
│   └── visualize.py             # EDA & visualization
│
├── models/
│   └── (saved models stored here)
│
├── outputs/
│   └── (charts and reports saved here)
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb   # Jupyter notebook walkthrough
│
├── app.py                       # Flask API for real-time prediction
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| ML Models | Random Forest, Logistic Regression, XGBoost |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | Flask REST API |
| Evaluation | ROC-AUC, F1-Score, Confusion Matrix |

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Shaikasif346/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python data/generate_data.py

# 4. Train the model
python src/train.py

# 5. Run Flask API
python app.py
```

---

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | **92%** | **0.91** | **0.96** |
| Logistic Regression | 84% | 0.83 | 0.89 |
| XGBoost | 91% | 0.90 | 0.95 |

---

## 🔌 API Usage

```bash
# Start the API
python app.py

# Send a prediction request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 65.5, "total_charges": 786.0,
       "contract": 0, "internet_service": 1, "tech_support": 0}'
```

**Response:**
```json
{
  "prediction": "Churn",
  "probability": 0.87,
  "confidence": "High"
}
```

---

## 👨‍💻 Author

**Shaik Kodivella Mahammad Asif**
- GitHub: [@Shaikasif346](https://github.com/Shaikasif346)
- LinkedIn: [sk-mahammadasif](https://www.linkedin.com/in/sk-mahammadasif-3a5a06339)
- Email: kodivellamahammad@gmail.com

---

## 📄 License
This project is licensed under the MIT License.
