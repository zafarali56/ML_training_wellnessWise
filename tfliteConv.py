import tensorflow as tf
import numpy as np

# Load the Keras model
model = tf.keras.models.load_model('enhanced_health_risk_model.keras')

# Create a concrete function from the Keras model
# Update the number in TensorSpec to match your model's input shape
@tf.function(input_signature=[tf.TensorSpec([None, 27], tf.float32)])
def model_function(input_tensor):
    return model(input_tensor)

concrete_func = model_function.get_concrete_function()

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open('enhanced_health_risk_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as 'enhanced_health_risk_model.tflite'")

# Optional: Test the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a sample input
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("TFLite model test output shape:", output_data.shape)
print("TFLite model test output:", output_data)
