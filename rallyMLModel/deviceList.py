import tensorflow as tf  # Import the TensorFlow library

# List all physical devices (CPUs, GPUs, TPUs) available to TensorFlow on this machine
for device in tf.config.list_physical_devices():
    print(device)  # Print the device information, e.g. type and name
