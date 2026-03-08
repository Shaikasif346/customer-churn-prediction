"""
Data Preprocessing & Feature Engineering Module
Handles all data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='data/customer_churn.csv'):
    """Load the dataset"""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """Basic EDA summary"""
    print("\n📊 Dataset Overview:")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n   Churn Distribution:")
    print(df['churn'].value_counts(normalize=True).map('{:.1%}'.format))
    print(f"\n   Numeric Summary:")
    print(df.describe().round(2))


def preprocess(df, target_col='churn', test_size=0.2, apply_smote=True):
    """
    Full preprocessing pipeline:
    - Drop irrelevant columns
    - Handle missing values
    - Feature engineering
    - Encoding
    - Scaling
    - Train/test split
    - SMOTE for class imbalance
    """

    df = df.copy()

    # Drop customer ID (not a feature)
    if 'customer_id' in df.columns:
        df.drop('customer_id', axis=1, inplace=True)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Feature Engineering
    df['charges_per_month_ratio'] = df['total_charges'] / (df['tenure'] + 1)
    df['is_new_customer'] = (df['tenure'] < 12).astype(int)
    df['has_support'] = ((df['tech_support'] == 1) | (df['online_security'] == 1)).astype(int)
    df['num_services'] = (df[['phone_service', 'multiple_lines',
                               'online_security', 'tech_support', 'streaming_tv']].sum(axis=1))

    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Apply SMOTE to handle class imbalance
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied — Training set balanced: {pd.Series(y_train).value_counts().to_dict()}")

    print(f"✅ Preprocessing complete:")
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Testing samples  : {X_test.shape[0]}")
    print(f"   Features         : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)


if __name__ == "__main__":
    df = load_data()
    explore_data(df)
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    print(f"\nFeatures used: {features}")
