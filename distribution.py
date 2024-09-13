import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('improved_wellness_dataset.csv')

# Separate features and targets
risk_columns = ['Diabetes_Risk', 'Cardiovascular_Disease_Risk', 'Hypertension_Risk', 'Obesity_Risk', 'Cancer_Risk']
X = df.drop(risk_columns, axis=1)
y = df[risk_columns]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model = keras.models.load_model('final_wellness_model.keras', compile=False)

# Compile the model (use the same custom loss function if you used one)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Make predictions on the entire test set
y_pred = model.predict(X_test)

# Plot histograms of predictions for each risk
plt.figure(figsize=(15, 10))
for i, col in enumerate(risk_columns):
    plt.subplot(2, 3, i+1)
    plt.hist(y_pred[:, i], bins=30)
    plt.title(f'{col} Prediction Distribution')
    plt.xlabel('Predicted Risk (Standardized)')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('prediction_distributions.png')
plt.close()

# Print summary statistics of predictions
for i, col in enumerate(risk_columns):
    print(f"\n{col} prediction statistics:")
    print(f"Mean: {y_pred[:, i].mean():.4f}")
    print(f"Std Dev: {y_pred[:, i].std():.4f}")
    print(f"Min: {y_pred[:, i].min():.4f}")
    print(f"Max: {y_pred[:, i].max():.4f}")
    print(f"10th percentile: {np.percentile(y_pred[:, i], 10):.4f}")
    print(f"90th percentile: {np.percentile(y_pred[:, i], 90):.4f}")

# Compare predictions with actual values
for i, col in enumerate(risk_columns):
    correlation = np.corrcoef(y_test[col], y_pred[:, i])[0, 1]
    print(f"\nCorrelation between actual and predicted {col}: {correlation:.4f}")

print("\nCheck the 'prediction_distributions.png' file for visualizations of the prediction distributions.")
