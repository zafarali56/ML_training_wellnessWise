import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('improved_wellness_dataset.csv')

# Analyze risk distributions
risk_columns = ['Diabetes_Risk', 'Cardiovascular_Disease_Risk', 'Hypertension_Risk', 'Obesity_Risk', 'Cancer_Risk']

plt.figure(figsize=(15, 10))
for i, col in enumerate(risk_columns, 1):
    plt.subplot(2, 3, i)
    plt.hist(df[col], bins=30)
    plt.title(f'{col} Distribution')
    plt.xlabel('Risk Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('risk_distributions.png')
plt.close()

# Print summary statistics
for col in risk_columns:
    print(f"\n{col} statistics:")
    print(df[col].describe())
    print(f"10th percentile: {df[col].quantile(0.1):.4f}")
    print(f"90th percentile: {df[col].quantile(0.9):.4f}")

# Calculate correlations between risks
correlation_matrix = df[risk_columns].corr()
print("\nRisk Correlation Matrix:")
print(correlation_matrix)
