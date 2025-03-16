#Use Case: Sequential data prediction (e.g., time series forecasting, text generation).
#Example: Predicting stock prices based on historical data.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np

# Generate synthetic data (e.g., sine wave for time series)
X = np.linspace(0, 100, 1000)
y = np.sin(X)

# Prepare data for RNN (reshape data into 3D [samples, timesteps, features])
X_data = np.reshape(X, (X.shape[0], 1, 1))

# Build an RNN model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(1, 1)),
    Dense(1)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_data, y, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(X_data)

# Output a portion of predictions
print(predictions[:5])
