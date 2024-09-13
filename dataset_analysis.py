import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv('improved_wellness_dataset.csv')

# 1. Basic Statistics
print("Dataset Shape:", df.shape)
print("\nBasic Statistics:")
print(df.describe())

# 2. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Risk Distributions
risk_columns = ['Diabetes_Risk', 'Cardiovascular_Disease_Risk', 'Hypertension_Risk', 'Obesity_Risk', 'Cancer_Risk']

plt.figure(figsize=(15, 10))
for i, col in enumerate(risk_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
    plt.xlabel('Risk Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('risk_distributions.png')
plt.close()

# 4. Correlation Analysis
corr_matrix = df.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# 5. Feature Importance (using correlation with risk factors)
feature_importance = corr_matrix[risk_columns].abs().mean().sort_values(ascending=False)
print("\nFeature Importance (based on correlation with risk factors):")
print(feature_importance)

# 6. Risk Factor Correlations
risk_corr = df[risk_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(risk_corr, annot=True, cmap='coolwarm')
plt.title('Risk Factor Correlations')
plt.savefig('risk_correlations.png')
plt.close()

# 7. Distribution of key features
key_features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Blood_Sugar', 'Cholesterol']
plt.figure(figsize=(15, 10))
for i, col in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('key_feature_distributions.png')
plt.close()

# 8. Check for class imbalance in risk factors
print("\nRisk Factor Class Balance:")
for col in risk_columns:
    low = (df[col] < -0.5).sum()
    medium = ((df[col] >= -0.5) & (df[col] <= 0.5)).sum()
    high = (df[col] > 0.5).sum()
    total = len(df)
    print(f"\n{col}:")
    print(f"Low Risk: {low} ({low/total*100:.2f}%)")
    print(f"Medium Risk: {medium} ({medium/total*100:.2f}%)")
    print(f"High Risk: {high} ({high/total*100:.2f}%)")

# 9. Outlier Detection
print("\nOutlier Detection (Z-score method):")
for col in df.columns:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers = (z_scores > 3).sum()
    if outliers > 0:
        print(f"{col}: {outliers} outliers detected")

print("\nAnalysis complete. Check the generated PNG files for visualizations.")
