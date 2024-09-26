import numpy as np
import pandas as pd
import tensorflow as tf

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

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="enhanced_health_risk_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create sample input data with 10 diverse patient profiles, replacing the 2nd patient with Firestore data
sample_input = pd.DataFrame({
    'Age':                          [25,   21,   70,   30,   55,   40,   60,   35,   50,   65],
    'Height':                       [180,  176,  170,  175,  168,  182,  160,  178,  172,  165],
    'Weight':                       [70,   63,   75,   65,   80,   95,   70,   85,   78,   72],
    'BMI':                          [21.6, 20.3, 26.0, 21.2, 28.3, 28.7, 27.3, 26.8, 26.4, 26.4],
    'Systolic_BP':                  [110,  110,  160,  115,  135,  145,  150,  120,  130,  155],
    'Diastolic_BP':                 [70,   71,   95,   75,   85,   95,   90,   80,   85,   95],
    'Heart_Rate':                   [65,   65,   75,   70,   72,   78,   68,   75,   70,   72],
    'Blood_Sugar':                  [80,   80,   110,  85,   120,  140,  100,  90,   110,  125],
    'Cholesterol':                  [170,  80,   200,  180,  220,  250,  190,  200,  210,  230],
    'Smoking':                      [0,    0,    1,    0,    1,    2,    0,    1,    0,    1],
    'Alcohol_Consumption':          [1,    1,    0,    2,    1,    3,    1,    2,    0,    1],
    'Physical_Activity':            [4,    1,    2,    3,    2,    1,    3,    4,    2,    1],
    'Diet_Quality':                 [4,    1,    3,    4,    3,    2,    3,    4,    3,    2],
    'Sleep_Hours':                  [8,    9,    6,    7,    6,    5,    7,    8,    6,    5],
    'Air_Quality_Index':            [30,   50,   80,   40,   90,   100,  70,   50,   80,   110],
    'Stress_Level':                 [1,    1,    3,    2,    3,    4,    2,    1,    3,    4],
    'Exposure_to_Pollutants':       [0,    1,    2,    1,    2,    3,    1,    0,    2,    3],
    'Access_to_Healthcare':         [4,    1,    2,    4,    3,    2,    3,    4,    2,    1],
    'Family_History_Diabetes':      [0,    0,    1,    0,    1,    1,    0,    0,    1,    1],
    'Family_History_Heart_Disease': [0,    0,    1,    0,    1,    0,    1,    0,    1,    1],
    'Family_History_Cancer':        [0,    0,    1,    0,    1,    0,    1,    0,    0,    1],
    'Previous_Surgeries':           [0,    0,    2,    0,    1,    0,    1,    0,    1,    2],
    'Chronic_Conditions':           [0,    0,    1,    0,    1,    1,    1,    0,    1,    2],
    'Gender_Female':                [0,    0,    0,    1,    1,    0,    1,    0,    1,    0],
    'Gender_Male':                  [1,    1,    1,    0,    0,    1,    0,    1,    0,    1]
})

# Scale the input data (use the same scaling method as in training)
scaled_input = (sample_input - sample_input.mean()) / sample_input.std()

# Convert to numpy array and ensure it's float32
input_data = scaled_input.values.astype(np.float32)

# Define risk categories
risk_categories = ['Diabetes', 'Cardiovascular Disease', 'Hypertension', 'Obesity', 'Cancer']

# Make predictions
for i in range(len(input_data)):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_data[i], axis=0))
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    print(f"\nPerson {i+1}:")
    print(sample_input.iloc[i])
    print("\nPredicted Risks:")
    for j, category in enumerate(risk_categories):
        risk_value = predictions[j]
        risk_class = classify_risk(risk_value)
        print(f"{category}: {risk_value:.4f} ({risk_class})")
    print()
