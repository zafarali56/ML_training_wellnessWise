import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Load the saved enhanced model
model = keras.models.load_model('enhanced_health_risk_model.keras')

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

# Create new sample input data with more varied health profiles
new_sample_input = pd.DataFrame({
    'Age':                          [25,   45,   70,   30,   55,   40,   60,   35,   50,   65],
    'Height':                       [180,  165,  170,  175,  168,  182,  160,  178,  172,  165],
    'Weight':                       [70,   90,   75,   65,   80,   95,   70,   85,   78,   72],
    'BMI':                          [21.6, 33.1, 26.0, 21.2, 28.3, 28.7, 27.3, 26.8, 26.4, 26.4],
    'Systolic_BP':                  [110,  140,  160,  115,  135,  145,  150,  120,  130,  155],
    'Diastolic_BP':                 [70,   90,   95,   75,   85,   95,   90,   80,   85,   95],
    'Heart_Rate':                   [65,   80,   75,   70,   72,   78,   68,   75,   70,   72],
    'Blood_Sugar':                  [80,   130,  110,  85,   120,  140,  100,  90,   110,  125],
    'Cholesterol':                  [170,  240,  200,  180,  220,  250,  190,  200,  210,  230],
    'Smoking':                      [0,    2,    1,    0,    1,    2,    0,    1,    0,    1],
    'Alcohol_Consumption':          [1,    3,    0,    2,    1,    3,    1,    2,    0,    1],
    'Physical_Activity':            [4,    1,    2,    3,    2,    1,    3,    4,    2,    1],
    'Diet_Quality':                 [4,    2,    3,    4,    3,    2,    3,    4,    3,    2],
    'Sleep_Hours':                  [8,    5,    6,    7,    6,    5,    7,    8,    6,    5],
    'Air_Quality_Index':            [30,   120,  80,   40,   90,   100,  70,   50,   80,   110],
    'Stress_Level':                 [1,    4,    3,    2,    3,    4,    2,    1,    3,    4],
    'Exposure_to_Pollutants':       [0,    3,    2,    1,    2,    3,    1,    0,    2,    3],
    'Access_to_Healthcare':         [4,    1,    2,    4,    3,    2,    3,    4,    2,    1],
    'Family_History_Diabetes':      [0,    1,    1,    0,    1,    1,    0,    0,    1,    1],
    'Family_History_Heart_Disease': [0,    1,    1,    0,    1,    0,    1,    0,    1,    1],
    'Family_History_Cancer':        [0,    0,    1,    0,    1,    0,    1,    0,    0,    1],
    'Previous_Surgeries':           [0,    1,    2,    0,    1,    0,    1,    0,    1,    2],
    'Chronic_Conditions':           [0,    2,    1,    0,    1,    1,    1,    0,    1,    2],
    'Gender_Female':                [0,    1,    0,    1,    1,    0,    1,    0,    1,    0],
    'Gender_Male':                  [1,    0,    1,    0,    0,    1,    0,    1,    0,    1]
})

# Print input shape
print(f"Input shape: {new_sample_input.shape}")

# Define risk categories
risk_categories = ['Diabetes', 'Cardiovascular Disease', 'Hypertension', 'Obesity', 'Cancer']

# Scale the input data
scaler = StandardScaler()
scaled_input = scaler.fit_transform(new_sample_input)

# Make predictions
try:
    predictions = model.predict(scaled_input)
    # Print results
    for i, (_, person_data) in enumerate(new_sample_input.iterrows()):
        print(f"\nPerson {i+1}:")
        print(person_data)
        print("\nPredicted Risks:")
        for j, category in enumerate(risk_categories):
            risk_value = predictions[i][j]
            risk_class = classify_risk(risk_value)
            print(f"{category}: {risk_value:.4f} ({risk_class})")
        print()
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Provided input shape: {scaled_input.shape}")
