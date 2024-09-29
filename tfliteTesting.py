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
    'Age':                          [18,   45,   65,   35,   55,   28,   50,   70,   40,   60,   22,   48,   62,   32,   58],
    'Height':                       [178,  165,  175,  180,  160,  170,  168,  172,  175,  162,  182,  167,  170,  178,  163],
    'Weight':                       [78,   80,   90,   75,   70,   65,   85,   88,   72,   78,   82,   76,   92,   70,   74],
    'BMI':                          [24.6, 29.4, 29.4, 23.1, 27.3, 22.5, 30.1, 29.7, 23.5, 29.7, 24.8, 27.3, 31.8, 22.1, 27.9],
    'Systolic_BP':                  [118,  140,  150,  120,  135,  110,  145,  155,  125,  142,  115,  138,  152,  122,  140],
    'Diastolic_BP':                 [79,   90,   95,   80,   85,   75,   92,   98,   82,   88,   78,   89,   96,   81,   87],
    'Heart_Rate':                   [67,   75,   80,   70,   72,   65,   78,   82,   72,   76,   68,   74,   79,   71,   73],
    'Blood_Sugar':                  [130,  130,  140,  95,   110,  85,   135,  145,  100,  120,  90,   125,  138,  98,   115],
    'Cholesterol':                  [250,  240,  220,  190,  210,  180,  230,  235,  200,  225,  195,  235,  228,  185,  215],
    'Triglycerides':                [150,  200,  180,  120,  150,  100,  190,  185,  130,  170,  110,  195,  175,  125,  160],
    'Waist_Circumference':          [100,  95,   100,  80,   90,   75,   98,   102,  85,   95,   88,   96,   101,  82,   93],
    'Smoking':                      [0,    1,    1,    0,    1,    0,    1,    0,    0,    1,    0,    1,    1,    0,    0],
    'Alcohol_Consumption':          [1,    3,    2,    1,    2,    2,    3,    1,    2,    2,    3,    2,    1,    1,    2],
    'Physical_Activity':            [0,    2,    1,    3,    2,    3,    1,    0,    2,    1,    3,    2,    1,    3,    2],
    'Diet_Quality':                 [0,    2,    2,    3,    3,    3,    1,    1,    2,    2,    3,    2,    1,    3,    2],
    'Sleep_Hours':                  [9,    6,    5,    7,    6,    8,    5,    6,    7,    6,    8,    6,    5,    7,    6],
    'Air_Quality_Index':            [150,  90,   100,  50,   70,   60,   110,  120,  80,   95,   70,   100,  115,  65,   85],
    'Stress_Level':                 [4,    4,    3,    2,    3,    2,    4,    3,    3,    3,    2,    4,    3,    2,    3],
    'Exposure_to_Pollutants':       [0,    3,    2,    1,    2,    1,    3,    2,    2,    2,    1,    3,    2,    1,    2],
    'Access_to_Healthcare':         [0,    2,    2,    3,    3,    3,    1,    1,    2,    2,    3,    2,    1,    3,    2],
    'Family_History_Diabetes':      [0,    1,    1,    0,    1,    0,    1,    1,    0,    1,    0,    1,    1,    0,    1],
    'Family_History_Heart_Disease': [0,    1,    1,    0,    1,    0,    1,    1,    1,    1,    0,    1,    1,    0,    0],
    'Family_History_Cancer':        [0,    0,    1,    0,    1,    0,    0,    1,    0,    1,    0,    1,    1,    0,    1],
    'Previous_Surgeries':           [0,    1,    2,    0,    1,    0,    1,    2,    0,    1,    0,    1,    2,    0,    1],
    'Chronic_Conditions':           [0,    1,    2,    0,    1,    0,    1,    2,    0,    1,    0,    1,    2,    0,    1],
    'Gender_Female':                [0,    1,    0,    1,    1,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1],
    'Gender_Male':                  [1,    0,    1,    0,    0,    0,    1,    0,    1,    0,    1,    0,    1,    0,    0]
})
# Calculate mean and standard deviation for each feature
feature_means = sample_input.mean()
feature_stds = sample_input.std()

print("Feature Means:")
print(feature_means.to_dict())
print("\nFeature Standard Deviations:")
print(feature_stds.to_dict())

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
