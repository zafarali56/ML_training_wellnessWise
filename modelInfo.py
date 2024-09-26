import numpy as np
import tensorflow as tf

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter("enhanced_health_risk_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def get_input_output_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def analyze_model(interpreter):
    input_details, output_details = get_input_output_details(interpreter)

    print("Input Details:")
    print(f"Shape: {input_details[0]['shape']}")
    print(f"Type: {input_details[0]['dtype']}")

    print("\nOutput Details:")
    print(f"Shape: {output_details[0]['shape']}")
    print(f"Type: {output_details[0]['dtype']}")

def normalize_input(input_data, feature_ranges):
    normalized = []
    for value, (min_val, max_val) in zip(input_data, feature_ranges):
        normalized.append((value - min_val) / (max_val - min_val))
    return normalized

def predict(interpreter, input_data):
    input_details, output_details = get_input_output_details(interpreter)

    interpreter.set_tensor(input_details[0]['index'], np.array([input_data], dtype=np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def adjust_output(output, adjustments, intercepts):
    adjusted = []
    for value, adj, intercept in zip(output, adjustments, intercepts):
        adjusted.append(min(max(value * adj + intercept, 0), 1))
    return adjusted

def main():
    model_path = "path/to/your/enhanced_health_risk_model.tflite"
    interpreter = load_tflite_model(model_path)

    analyze_model(interpreter)

    # Define feature ranges for normalization
    feature_ranges = [
        (18, 100),  # Age
        (140, 220),  # Height
        (40, 200),  # Weight
        (15, 50),  # BMI
        (80, 200),  # Systolic BP
        (50, 130),  # Diastolic BP
        (40, 120),  # Heart Rate
        (70, 300),  # Blood Sugar
        (100, 300),  # Cholesterol
        (0, 3),  # Smoking
        (0, 4),  # Alcohol Consumption
        (0, 4),  # Physical Activity
        (0, 4),  # Diet Quality
        (4, 12),  # Sleep Hours
        (0, 500),  # Air Quality Index
        (0, 4),  # Stress Level
        (0, 3),  # Exposure to Pollutants
        (0, 4),  # Access to Healthcare
        (0, 1),  # Family History Diabetes
        (0, 1),  # Family History Heart Disease
        (0, 1),  # Family History Cancer
        (0, 3),  # Previous Surgeries
        (0, 3),  # Chronic Conditions
        (0, 1),  # Gender Female
        (0, 1),  # Gender Male
    ]

    # Test cases
    test_cases = [
        [25, 180, 70, 21.6, 110, 70, 65, 80, 170, 0, 1, 4, 4, 8, 30, 1, 0, 4, 0, 0, 0, 0, 0, 0, 1],  # Healthy
        [45, 165, 90, 33.1, 140, 90, 80, 130, 240, 2, 3, 1, 2, 5, 120, 4, 3, 1, 1, 1, 0, 1, 2, 1, 0],  # High risk
        [70, 170, 75, 26.0, 160, 95, 75, 110, 200, 1, 0, 2, 3, 6, 80, 3, 2, 2, 1, 1, 1, 2, 1, 0, 1],  # Elderly
    ]

    # Adjustments and intercepts
    adjustments = [1.2, 1.2, 1.1, 1.2, 1.0]
    intercepts = [-0.1, -0.1, -0.05, -0.1, -0.05]

    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        normalized_input = normalize_input(case, feature_ranges)
        raw_output = predict(interpreter, normalized_input)
        adjusted_output = adjust_output(raw_output, adjustments, intercepts)

        print("Raw input:", case)
        print("Normalized input:", normalized_input)
        print("Raw output:", raw_output)
        print("Adjusted output:", adjusted_output)

if __name__ == "__main__":
    main()
