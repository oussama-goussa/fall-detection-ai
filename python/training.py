# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Function to prepare and resample data
def split_resampling(data, seed=1218):
    # Prepare the data
    new_data = data.drop("label", axis=1).to_numpy().reshape(-1, 400, 6)
    labels = np.array(data.label.iloc[np.arange(0, data.shape[0], 400)])

    # Apply SMOTE for balancing the dataset
    sm = SMOTE(random_state=seed)
    X, y = sm.fit_resample(new_data.reshape(-1, 400*6), labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Scale the training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 400*6))

    # Scale the test data (using the same scaler fitted on the training data)
    X_test = scaler.transform(X_test.reshape(-1, 400*6))

    # Reshape back for model input
    X_train = X_train.reshape(-1, 400, 6)
    X_test = X_test.reshape(-1, 400, 6)

    # Define the label encoder
    le = LabelEncoder()
    le.fit(np.unique(y_train))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # Convert labels to categorical (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=7)

    return X_train, X_test, y_train, y_test, le, X_train.shape[1:]

# Load the data
data = pd.read_csv("./data/acc_gyr.csv")

# Split the dataset and prepare the data
X_train, X_test, y_train, y_test, le, input_shape = split_resampling(data)

# Save the label encoder for later use in prediction
joblib.dump(le, './models/label_encoder.pkl')

# Print the shape of the training data
print(X_train.shape)

# Build the flexible CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(None, 6)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # Use 'softmax' for multi-class classification
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
model.evaluate(X_test, y_test)

# Save the trained model
model.save('./models/acc_gyr_model.h5')
print("Model saved as 'acc_gyr_model.h5'")