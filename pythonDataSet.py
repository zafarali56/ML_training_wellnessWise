import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_enhanced_dataset(num_samples=50000):
    np.random.seed(42)

    # Generate basic user information
    age = np.random.randint(18, 80, num_samples)
    gender = np.random.choice(['Male', 'Female'], num_samples)
    height = np.random.normal(170, 10, num_samples)
    weight = np.random.normal(70, 15, num_samples)

    # Generate health parameters
    bmi = weight / ((height / 100) ** 2)
    systolic_bp = np.random.normal(120, 15, num_samples) + (age - 50) * 0.5 + (bmi - 25) * 2
    diastolic_bp = np.random.normal(80, 10, num_samples) + (age - 50) * 0.3 + (bmi - 25) * 1.5
    heart_rate = np.random.normal(70, 10, num_samples) + (age - 50) * 0.2
    blood_sugar = np.random.normal(100, 20, num_samples) + (bmi - 25) * 3
    cholesterol = np.random.normal(200, 40, num_samples) + (age - 50) * 1 + (bmi - 25) * 4

    # New features
    triglycerides = np.random.normal(150, 50, num_samples) + (bmi - 25) * 5 + (age - 50) * 0.5
    waist_circumference = np.random.normal(90, 15, num_samples) + (bmi - 25) * 2

    # Generate lifestyle habits
    smoking = np.random.choice([0, 1, 2, 3], num_samples, p=[0.6, 0.2, 0.15, 0.05])
    alcohol_consumption = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    physical_activity = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    diet_quality = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    sleep_hours = np.clip(np.random.normal(7, 1.5, num_samples), 4, 12)

    # Generate environmental factors
    air_quality_index = np.clip(np.random.normal(50, 20, num_samples), 0, 500)
    stress_level = np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    exposure_to_pollutants = np.random.choice([0, 1, 2, 3], num_samples, p=[0.4, 0.3, 0.2, 0.1])
    access_to_healthcare = np.random.choice([0, 1, 2, 3], num_samples, p=[0.1, 0.2, 0.4, 0.3])

    # Generate medical history
    family_history_diabetes = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    family_history_heart_disease = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    family_history_cancer = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    previous_surgeries = np.random.choice([0, 1, 2, 3], num_samples, p=[0.6, 0.2, 0.15, 0.05])
    chronic_conditions = np.random.choice([0, 1, 2, 3], num_samples, p=[0.7, 0.2, 0.08, 0.02])

    # Generate health risks
    def calculate_risk(base, specific_factors, noise_factor=0.05):
        # Combine base risk and specific factors
        combined_risk = base + sum(specific_factors)

        # Apply sigmoid function to create more mid-range values
        sigmoid_risk = 1 / (1 + np.exp(-5 * (combined_risk - 0.5)))

        # Add some noise to create more variation
        noisy_risk = sigmoid_risk + np.random.normal(0, noise_factor, len(base))

        # Clip values to ensure they're between 0 and 1
        return np.clip(noisy_risk, 0, 1)

    base_risk = (
        (age / 100) * 0.2 +  # Reduced weight
        ((bmi - 25) / 10) * 0.15 +  # Reduced weight
        (smoking / 3) * 0.1 +  # Reduced weight
        (alcohol_consumption / 4) * 0.08 +  # Reduced weight
        ((4 - physical_activity) / 4) * 0.08 +
        ((4 - diet_quality) / 4) * 0.08 +
        (np.abs(sleep_hours - 7) / 3) * 0.05 +
        (air_quality_index / 500) * 0.05 +
        (stress_level / 4) * 0.05 +
        (exposure_to_pollutants / 3) * 0.05 +
        ((3 - access_to_healthcare) / 3) * 0.05 +
        ((triglycerides - 150) / 100) * 0.08 +
        ((waist_circumference - 90) / 20) * 0.08
    ) / 2  # Divide by 2 to reduce overall base risk

    # Adjust risk calculations for each health condition
    diabetes_risk = calculate_risk(base_risk, [
        ((blood_sugar - 100) / 100) * 0.3,
        ((bmi - 25) / 10) * 0.2,
        ((4 - physical_activity) / 4) * 0.15,
        ((4 - diet_quality) / 4) * 0.1,
        family_history_diabetes * 0.15,
        ((triglycerides - 150) / 100) * 0.15,
        ((waist_circumference - 90) / 20) * 0.15
    ])

    cardiovascular_disease_risk = calculate_risk(base_risk, [
        ((systolic_bp - 120) / 50) * 0.25,
        ((cholesterol - 200) / 100) * 0.25,
        (smoking / 3) * 0.15,
        ((4 - physical_activity) / 4) * 0.15,
        family_history_heart_disease * 0.15,
        ((triglycerides - 150) / 100) * 0.15,
        ((waist_circumference - 90) / 20) * 0.1
    ])

    hypertension_risk = calculate_risk(base_risk, [
        ((systolic_bp - 120) / 50) * 0.3,
        ((diastolic_bp - 80) / 40) * 0.25,
        (stress_level / 4) * 0.15,
        (alcohol_consumption / 4) * 0.1,
        ((waist_circumference - 90) / 20) * 0.15
    ])

    obesity_risk = calculate_risk(base_risk, [
        ((bmi - 25) / 10) * 0.4,
        ((4 - physical_activity) / 4) * 0.25,
        ((4 - diet_quality) / 4) * 0.2,
        ((waist_circumference - 90) / 20) * 0.25
    ])

    cancer_risk = calculate_risk(base_risk, [
        (smoking / 3) * 0.25,
        (alcohol_consumption / 4) * 0.15,
        ((age - 50) / 30) * 0.15,
        ((4 - diet_quality) / 4) * 0.15,
        (air_quality_index / 500) * 0.1,
        family_history_cancer * 0.15,
        ((waist_circumference - 90) / 20) * 0.1
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
        'Triglycerides': triglycerides,  # New feature
        'Waist_Circumference': waist_circumference,  # New feature
        'Smoking': smoking,
        'Alcohol_Consumption': alcohol_consumption,
        'Physical_Activity': physical_activity,
        'Diet_Quality': diet_quality,
        'Sleep_Hours': sleep_hours,
        'Air_Quality_Index': air_quality_index,
        'Stress_Level': stress_level,
        'Exposure_to_Pollutants': exposure_to_pollutants,
        'Access_to_Healthcare': access_to_healthcare,
        'Family_History_Diabetes': family_history_diabetes,
        'Family_History_Heart_Disease': family_history_heart_disease,
        'Family_History_Cancer': family_history_cancer,
        'Previous_Surgeries': previous_surgeries,
        'Chronic_Conditions': chronic_conditions,
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

def save_dataset(df, filename='enhanced_wellness_dataset.csv'):
    df.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_enhanced_dataset()

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
