import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns


# Load the dataset
df = pd.read_csv('enhanced_wellness_dataset.csv')

# Separate features and target variables
X = df.drop(['Diabetes_Risk', 'Cardiovascular_Disease_Risk', 'Hypertension_Risk', 'Obesity_Risk', 'Cancer_Risk'], axis=1)
y = df[['Diabetes_Risk', 'Cardiovascular_Disease_Risk', 'Hypertension_Risk', 'Obesity_Risk', 'Cancer_Risk']]

# Normalize target variables to 0-1 range
y = (y - y.min()) / (y.max() - y.min())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(5, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build the model
model = build_model(X_train_scaled.shape[1])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    verbose=1
)
# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Mean Absolute Error: {test_mae}")


# Save the model
model.save('enhanced_health_risk_model.keras')
print("Model saved as 'enhanced_health_risk_model.keras'")



# Make predictions
predictions = model.predict(X_test_scaled)

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

# Analyze prediction
risk_categories = ['Diabetes', 'Cardiovascular Disease', 'Hypertension', 'Obesity', 'Cancer']

for i, category in enumerate(risk_categories):
    print(f"\n{category} Risk Distribution:")
    classified_risks = [classify_risk(pred) for pred in predictions[:, i]]
    risk_counts = pd.Series(classified_risks).value_counts().sort_index()
    print(risk_counts)
    print(f"Mean predicted risk: {predictions[:, i].mean():.4f}")
    print(f"Actual mean risk: {y_test.iloc[:, i].mean():.4f}")
    print(f"MAE: {mean_absolute_error(y_test.iloc[:, i], predictions[:, i]):.4f}")
    print(f"MSE: {mean_squared_error(y_test.iloc[:, i], predictions[:, i]):.4f}")
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Plot predicted vs actual risks
plt.figure(figsize=(15, 10))
for i, category in enumerate(risk_categories):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=y_test.iloc[:, i], y=predictions[:, i])
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Risk')
    plt.ylabel('Predicted Risk')
    plt.title(f'{category} Risk')

plt.tight_layout()
plt.show()

# Feature importance analysis
feature_importance = np.abs(model.layers[-1].get_weights()[0])
feature_importance = np.mean(feature_importance, axis=1)
feature_names = X.columns.tolist()

# Ensure feature_importance and feature_names have the same length
min_length = min(len(feature_importance), len(feature_names))
feature_importance = feature_importance[:min_length]
feature_names = feature_names[:min_length]

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Mean Absolute Weight')
plt.tight_layout()
plt.show()
