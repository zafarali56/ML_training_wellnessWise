import tensorflow as tf
print("TensorFlow version:", tf.__version__)

print("Keras version (from TensorFlow):", tf.keras.__version__)

try:
    import keras
    print("Keras version (standalone):", keras.__version__)
except ImportError:
    print("Standalone Keras is not installed.")
