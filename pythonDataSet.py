import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_dataset(num_samples=50000):
    np.random.seed(42)

    # Generate basic user information
    age = np.random.randint(18, 80, num_samples)
    gender = np.random.choice(['Male', 'Female'], num_samples)
    height = np.random.normal(170, 10, num_samples)
    weight = np.random.normal(70, 15, num_samples)

    # Generate health parameters with more realistic correlations
    bmi = weight / ((height / 100) ** 2)
    systolic_bp = np.random.normal(120, 15, num_samples) + (age - 50) * 0.5 + (bmi - 25) * 2
    diastolic_bp = np.random.normal(80, 10, num_samples) + (age - 50) * 0.3 + (bmi - 25) * 1.5
    heart_rate = np.random.normal(70, 10, num_samples) + (age - 50) * 0.2
    blood_sugar = np.random.normal(100, 20, num_samples) + (bmi - 25) * 3
    cholesterol = np.random.normal(200, 40, num_samples) + (age - 50) * 1 + (bmi - 25) * 4

    # Generate lifestyle habits with more granularity
    smoking = np.random.choice([0, 1, 2, 3], num_samples, p=[0.6, 0.2, 0.15, 0.05])  # Non-smoker, Occasional, Regular, Heavy
    alcohol_consumption = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])  # None, Light, Moderate, Heavy, Very Heavy
    physical_activity = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])  # Sedentary, Light, Moderate, Active, Very Active
    diet_quality = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])  # Poor, Fair, Average, Good, Excellent
    sleep_hours = np.clip(np.random.normal(7, 1.5, num_samples), 4, 12)

    # Generate environmental factors
    air_quality_index = np.clip(np.random.normal(50, 20, num_samples), 0, 500)
    stress_level = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])  # Very Low, Low, Moderate, High, Very High

    # Generate health risks with more realistic correlations and ensure balanced distribution
    def calculate_risk(base, specific_factors, noise_factor=0.1):
        risk = base + sum(specific_factors) + np.random.normal(0, noise_factor, num_samples)
        return np.clip(risk, 0, 1)  # Ensure non-negative and max of 1

    base_risk = (
        (age / 100) * 0.3 +
        ((bmi - 25) / 10) * 0.2 +
        (smoking / 3) * 0.15 +
        (alcohol_consumption / 4) * 0.1 +
        ((4 - physical_activity) / 4) * 0.1 +
        ((4 - diet_quality) / 4) * 0.1 +
        (np.abs(sleep_hours - 7) / 3) * 0.05 +
        (air_quality_index / 500) * 0.05 +
        (stress_level / 4) * 0.05
    )

    diabetes_risk = calculate_risk(base_risk, [
        ((blood_sugar - 100) / 100) * 0.4,
        ((bmi - 25) / 10) * 0.3,
        ((4 - physical_activity) / 4) * 0.2,
        ((4 - diet_quality) / 4) * 0.1
    ])

    cardiovascular_disease_risk = calculate_risk(base_risk, [
        ((systolic_bp - 120) / 50) * 0.3,
        ((cholesterol - 200) / 100) * 0.3,
        (smoking / 3) * 0.2,
        ((4 - physical_activity) / 4) * 0.2
    ])

    hypertension_risk = calculate_risk(base_risk, [
        ((systolic_bp - 120) / 50) * 0.4,
        ((diastolic_bp - 80) / 40) * 0.3,
        (stress_level / 4) * 0.2,
        (alcohol_consumption / 4) * 0.1
    ])

    obesity_risk = calculate_risk(base_risk, [
        ((bmi - 25) / 10) * 0.5,
        ((4 - physical_activity) / 4) * 0.3,
        ((4 - diet_quality) / 4) * 0.2
    ])

    cancer_risk = calculate_risk(base_risk, [
        (smoking / 3) * 0.3,
        (alcohol_consumption / 4) * 0.2,
        ((age - 50) / 30) * 0.2,
        ((4 - diet_quality) / 4) * 0.2,
        (air_quality_index / 500) * 0.1
    ])

    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'BMI': bmi,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'Heart_Rate': heart_rate,
        'Blood_Sugar': blood_sugar,
        'Cholesterol': cholesterol,
        'Smoking': smoking,
        'Alcohol_Consumption': alcohol_consumption,
        'Physical_Activity': physical_activity,
        'Diet_Quality': diet_quality,
        'Sleep_Hours': sleep_hours,
        'Air_Quality_Index': air_quality_index,
        'Stress_Level': stress_level,
        'Diabetes_Risk': diabetes_risk,
        'Cardiovascular_Disease_Risk': cardiovascular_disease_risk,
        'Hypertension_Risk': hypertension_risk,
        'Obesity_Risk': obesity_risk,
        'Cancer_Risk': cancer_risk
    })

    return df

def preprocess_data(df):
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Gender'])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = df_encoded.select_dtypes(include=[np.number]).columns
    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

    return df_encoded

def save_dataset(df, filename='improved_wellness_dataset.csv'):
    df.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_dataset()

    # Preprocess data
    preprocessed_dataset = preprocess_data(dataset)

    # Save dataset
    save_dataset(preprocessed_dataset)

    print("Dataset statistics:")
    print(preprocessed_dataset.describe())

    # Print risk distribution
    risk_columns = ['Diabetes_Risk', 'Cardiovascular_Disease_Risk', 'Hypertension_Risk', 'Obesity_Risk', 'Cancer_Risk']
    for col in risk_columns:
        stable = (dataset[col] < 0.2).sum()
        mild = ((dataset[col] >= 0.2) & (dataset[col] < 0.4)).sum()
        moderate = ((dataset[col] >= 0.4) & (dataset[col] < 0.6)).sum()
        severe = ((dataset[col] >= 0.6) & (dataset[col] < 0.8)).sum()
        critical = (dataset[col] >= 0.8).sum()
        print(f"\n{col} distribution:")
        print(f"Stable: {stable/len(dataset)*100:.2f}%")
        print(f"Mild: {mild/len(dataset)*100:.2f}%")
        print(f"Moderate: {moderate/len(dataset)*100:.2f}%")
        print(f"Severe: {severe/len(dataset)*100:.2f}%")
        print(f"Critical: {critical/len(dataset)*100:.2f}%")
