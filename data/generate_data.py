"""
Customer Churn Dataset Generator
Generates realistic synthetic customer data for churn prediction
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_churn_dataset(n_samples=10000):
    """Generate a realistic customer churn dataset"""

    # Customer demographics
    tenure = np.random.randint(1, 72, n_samples)
    age = np.random.randint(18, 75, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male

    # Service features
    internet_service = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.5, 0.3])
    phone_service = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    multiple_lines = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    online_security = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    tech_support = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    streaming_tv = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])

    # Contract & billing
    contract = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.25, 0.25])  # 0=Monthly, 1=1yr, 2=2yr
    paperless_billing = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    payment_method = np.random.choice([0, 1, 2, 3], n_samples)

    # Charges
    monthly_charges = np.round(20 + (internet_service * 20) + (multiple_lines * 10) +
                               (streaming_tv * 10) + np.random.normal(0, 5, n_samples), 2)
    monthly_charges = np.clip(monthly_charges, 20, 120)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, n_samples), 2)
    total_charges = np.clip(total_charges, 0, None)

    # Churn logic (realistic probabilities)
    churn_prob = (
        0.05 +
        (contract == 0) * 0.25 +          # Monthly contract → higher churn
        (tenure < 12) * 0.20 +             # New customers → higher churn
        (tech_support == 0) * 0.10 +       # No tech support → higher churn
        (online_security == 0) * 0.10 +    # No security → higher churn
        (monthly_charges > 80) * 0.10 +    # High charges → higher churn
        (internet_service == 2) * 0.05     # Fiber optic → slightly higher
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob, n_samples)

    df = pd.DataFrame({
        'customer_id': [f'CUST{str(i).zfill(5)}' for i in range(n_samples)],
        'age': age,
        'gender': gender,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/customer_churn.csv', index=False)
    print(f"✅ Dataset generated: {n_samples} records")
    print(f"   Churn Rate: {churn.mean():.1%}")
    print(f"   Saved to: data/customer_churn.csv")
    return df

if __name__ == "__main__":
    df = generate_churn_dataset(10000)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nChurn Distribution:\n{df['churn'].value_counts()}")
