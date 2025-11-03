import joblib
import numpy as np
import tensorflow as tf
import serial

# Load the model, scaler, and label encoder
model = tf.keras.models.load_model('./models/acc_gyr_model.h5')
le = joblib.load('./models/label_encoder.pkl')

# Set up serial connection
ser = serial.Serial('COM2', 9600, timeout=1)

# Print message to indicate waiting for data
print("Waiting for data to start the prediction...")

try:
    while True:
        # Read data from Arduino (CSV format)
        line = ser.readline().decode('utf-8').strip()

        if line:
            values = line.split(',')
            if len(values) == 6:  # Ensure there are exactly 6 values
                try:
                    # Parse the values into float
                    acc_x, acc_y, acc_z = float(values[0]), float(values[1]), float(values[2])
                    gyro_x, gyro_y, gyro_z = float(values[3]), float(values[4]), float(values[5])

                    # Prepare the input data (reshape to match model's expected input)
                    input_data = np.array([[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]])

                    # Reshape the input data (model expects a 3D input shape: (samples, timesteps, features))
                    input_data = np.tile(input_data, (400, 1)).reshape(1, 400, 6)

                    # Make prediction using the trained model
                    prediction = model.predict(input_data)

                    # Get the predicted class
                    predicted_class = np.argmax(prediction, axis=1)[0]

                    # Convert predicted class back to original label
                    predicted_label = le.inverse_transform([predicted_class])[0]

                    # Output the result
                    print(f"Predicted Label: {predicted_label}")

                    # Envoi via le port série
                    ser.write(predicted_label.encode())  # Assurez-vous de bien ajouter une nouvelle ligne

                except ValueError:
                    print("Error converting values to float.")
            else:
                print(f"Skipping incomplete line: {line}")  # Log the issue

except KeyboardInterrupt:
    print("Program interrupted. Closing the serial connection...")
    ser.close()
