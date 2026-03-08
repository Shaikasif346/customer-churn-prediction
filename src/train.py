"""
Model Training & Evaluation Module
Trains Random Forest, Logistic Regression, and XGBoost models
Evaluates with accuracy, F1-score, ROC-AUC, and confusion matrix
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve)
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, preprocess

import warnings
warnings.filterwarnings('ignore')


def get_models():
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=42
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    return models


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return {'accuracy': acc, 'f1': f1, 'roc_auc': roc, 'y_pred': y_pred, 'y_prob': y_prob}


def plot_results(models_results, X_test, y_test, feature_names, best_model):
    os.makedirs('outputs', exist_ok=True)

    # 1. ROC Curves
    plt.figure(figsize=(10, 6))
    for name, res in models_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={res['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=150)
    plt.close()
    print("✅ Saved: outputs/roc_curves.png")

    # 2. Confusion Matrix (best model)
    best_name = max(models_results, key=lambda x: models_results[x]['accuracy'])
    cm = confusion_matrix(y_test, models_results[best_name]['y_pred'])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'], linewidths=1)
    plt.title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print("✅ Saved: outputs/confusion_matrix.png")

    # 3. Feature Importance (Random Forest)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in indices[::-1]],
                 importances[indices[::-1]], color='steelblue', edgecolor='white')
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 15 Feature Importances — Random Forest', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=150)
        plt.close()
        print("✅ Saved: outputs/feature_importance.png")

    # 4. Model Comparison Bar Chart
    model_names = list(models_results.keys())
    metrics = ['accuracy', 'f1', 'roc_auc']
    labels = ['Accuracy', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [models_results[m][metric] for m in model_names]
        bars = ax.bar(x + i*width, vals, width, label=label, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150)
    plt.close()
    print("✅ Saved: outputs/model_comparison.png")


def train_and_evaluate():
    print("🚀 Starting Customer Churn Prediction Pipeline...\n")

    # Load & preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    models = get_models()
    results = {}
    trained_models = {}

    # Train all models
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test, name)
        trained_models[name] = model

    # Best model
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_name]
    print(f"\n🏆 Best Model: {best_name} (Accuracy: {results[best_name]['accuracy']*100:.2f}%)")

    # Save best model
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler,
                     'features': feature_names, 'name': best_name}, f)
    print(f"✅ Model saved: models/best_model.pkl")

    # Plot results
    print("\n📊 Generating visualizations...")
    plot_results(results, X_test, y_test, feature_names, best_model)

    print("\n✅ Training pipeline complete!")
    return best_model, results


if __name__ == "__main__":
    train_and_evaluate()
