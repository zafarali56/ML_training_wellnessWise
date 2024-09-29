import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="enhanced_health_risk_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Use the same input as Person 5 from your original script
input_data = np.array([[0.6957011, -1.264911, -0.6238503, 0.2996196, 0.2504897, 0.10397505, -0.07149728, -0.08667028, 0.14804664, 0.0, 0.19289714, 0.23904571, 0.23904578, -0.35082328, 0.23904578, -0.35082328, 0.069843024, 0.35082328, 0.35082316, -0.95618284, 0.7302967, 0.7302967, 1.0954452, 0.23904571, 0.23904571, -1.0954452, 1.0954452]], dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("TFLite Model Output:", output_data[0])
