"""
EDA & Visualization Module
Generates comprehensive exploratory data analysis charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs('outputs', exist_ok=True)


def plot_churn_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count
    counts = df['churn'].value_counts()
    axes[0].bar(['No Churn', 'Churn'], counts.values,
                color=['#2196F3', '#F44336'], edgecolor='white', linewidth=1.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontsize=12, fontweight='bold')
    axes[0].set_title('Churn Count Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Count')

    # Pie
    axes[1].pie(counts.values, labels=['No Churn', 'Churn'],
                autopct='%1.1f%%', colors=['#2196F3', '#F44336'],
                startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[1].set_title('Churn Percentage', fontsize=13, fontweight='bold')

    plt.suptitle('Customer Churn Distribution', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/churn_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: outputs/churn_distribution.png")


def plot_numerical_features(df):
    num_cols = ['tenure', 'monthly_charges', 'total_charges', 'age']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for churn_val, color, label in [(0, '#2196F3', 'No Churn'), (1, '#F44336', 'Churn')]:
            data = df[df['churn'] == churn_val][col]
            axes[i].hist(data, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
        axes[i].set_title(f'{col.replace("_", " ").title()} by Churn', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col.replace("_", " ").title())
        axes[i].set_ylabel('Count')
        axes[i].legend()

    plt.suptitle('Numerical Features vs Churn', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/numerical_features.png', dpi=150)
    plt.close()
    print("✅ Saved: outputs/numerical_features.png")


def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=150)
    plt.close()
    print("✅ Saved: outputs/correlation_heatmap.png")


def plot_churn_by_contract(df):
    contract_map = {0: 'Month-to-Month', 1: 'One Year', 2: 'Two Year'}
    df['contract_type'] = df['contract'].map(contract_map)
    churn_rate = df.groupby('contract_type')['churn'].mean().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(churn_rate.index, churn_rate.values * 100,
                   color=['#F44336', '#FF9800', '#4CAF50'], edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, churn_rate.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
    plt.title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
    plt.ylabel('Churn Rate (%)')
    plt.xlabel('Contract Type')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/churn_by_contract.png', dpi=150)
    plt.close()
    print("✅ Saved: outputs/churn_by_contract.png")


def run_full_eda(filepath='data/customer_churn.csv'):
    print("📊 Running Full EDA...\n")
    df = pd.read_csv(filepath)

    plot_churn_distribution(df)
    plot_numerical_features(df)
    plot_correlation_heatmap(df)
    plot_churn_by_contract(df)

    print("\n✅ EDA Complete! All charts saved to outputs/")


if __name__ == "__main__":
    run_full_eda()
