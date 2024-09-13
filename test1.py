import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = keras.models.load_model('health_risk_model.keras')

# Function to classify risk levels
def classify_risk(prediction):
    if prediction < 0.2:
        return "Stable"
    elif prediction < 0.4:
        return "Mild"
    elif prediction < 0.6:
        return "Moderate"
    elif prediction < 0.8:
        return "Severe"
    else:
        return "Critical"

# Create sample input data with all 18 features
sample_input = pd.DataFrame({
    'Age': [35, 50, 65],
    'BMI': [22.5, 28.0, 32.5],
    'Blood_Pressure_Systolic': [120, 135, 150],
    'Blood_Pressure_Diastolic': [80, 85, 95],
    'Cholesterol': [180, 220, 260],
    'Glucose': [85, 100, 120],
    'Smoking': [0, 1, 1],
    'Alcohol_Consumption': [1, 2, 3],
    'Physical_Activity': [3, 2, 1],
    'Family_History_Diabetes': [0, 1, 1],
    'Family_History_Heart_Disease': [0, 0, 1],
    'Family_History_Cancer': [0, 1, 1],
    'Diet_Quality': [4, 3, 2],
    'Sleep_Hours': [7, 6, 5],
    'Stress_Level': [2, 3, 4],
    'Gender': [0, 1, 0],  # Assuming 0 for female, 1 for male
    'Waist_Circumference': [80, 95, 110],
    'Triglycerides': [120, 150, 200]
})

# Define risk categories
risk_categories = ['Diabetes', 'Cardiovascular Disease', 'Hypertension', 'Obesity', 'Cancer']

# Scale the input data
scaler = StandardScaler()
scaled_input = scaler.fit_transform(sample_input)

# Make predictions
predictions = model.predict(scaled_input)

# Print results
for i, (_, person_data) in enumerate(sample_input.iterrows()):
    print(f"\nPerson {i+1}:")
    print(person_data)
    print("\nPredicted Risks:")
    for j, category in enumerate(risk_categories):
        risk_value = predictions[i][j]
        risk_class = classify_risk(risk_value)
        print(f"{category}: {risk_value:.4f} ({risk_class})")
    print()
