from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from xai_module import analyze_reconstruction_error 
import os

# --- Configuration ---
TIME_STEPS = 10
TRAIN_SPLIT_PERCENT = 0.8
FILENAME = 'synthetic_sensor_data.csv'
MODEL_PATH = 'vae_model.h5'

app = Flask(__name__)

# --- Global Variables for Model and Data ---
vae_model = None
data_df = None
scaler = None
anomaly_threshold = 0.0  # Will be calculated after loading data
SENSOR_NAMES = []

def load_dependencies():
    """
    Loads the trained VAE model, the dataset, and prepares the scaler.
    This function runs once when the Flask app starts.
    """
    global vae_model, data_df, scaler, anomaly_threshold, SENSOR_NAMES
    
    print("Loading dependencies for Flask app...")

    # 1. Load the dataset
    if not os.path.exists(FILENAME):
        print(f"\nERROR: {FILENAME} not found. Please run eda.py first.\n")
        return False
    data_df = pd.read_csv(FILENAME, index_col='Timestamp', parse_dates=True)
    SENSOR_NAMES = [col for col in data_df.columns]
    
    # 2. Load the trained VAE model
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: {MODEL_PATH} not found. Please run vae_model.py to train and save the model.\n")
        return False
    vae_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 3. Prepare and fit the scaler ONLY on the training data
    # This is crucial to prevent data leakage from the test/anomaly set
    train_size = int(len(data_df) * TRAIN_SPLIT_PERCENT)
    train_df = data_df.iloc[:train_size]
    
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    
    # 4. Calculate the anomaly threshold from the model's performance on normal data
    # Create training sequences to find the max reconstruction error
    train_scaled = scaler.transform(train_df)
    X_train_seq = []
    for i in range(len(train_scaled) - TIME_STEPS):
        X_train_seq.append(train_scaled[i:(i + TIME_STEPS)])
    X_train_seq = np.array(X_train_seq)
    
    X_train_pred = vae_model.predict(X_train_seq, verbose=0)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train_seq), axis=(1, 2))
    anomaly_threshold = np.max(train_mae_loss) * 1.1 # 10% buffer
    
    print("--- Dependencies loaded successfully ---")
    print(f"Dataset Shape: {data_df.shape}")
    print(f"Model: {MODEL_PATH} loaded.")
    print(f"Anomaly threshold calculated: {anomaly_threshold:.4f}")
    print("------------------------------------")
    return True

# --- App Initialization ---
is_loaded = load_dependencies()
current_index = TIME_STEPS # Start after the first sequence window

# --- API Endpoints ---

@app.route('/')
def index():
    """Renders the main dashboard page."""
    # The templates folder should be in the same directory as this script
    return render_template('index.html', sensor_names=SENSOR_NAMES)

@app.route('/api/sensor-data')
def get_sensor_data():
    """API endpoint to stream the next row of sensor data."""
    global current_index
    if not is_loaded:
        return jsonify({"error": "App dependencies not loaded."}), 500

    if current_index >= len(data_df):
        current_index = TIME_STEPS  # Loop back

    data_point = data_df.iloc[current_index]
    response = {
        'timestamp': data_point.name.strftime('%Y-%m-%d %H:%M:%S'),
        'sensors': {name: val for name, val in zip(SENSOR_NAMES, data_point.values)}
    }
    
    current_index += 1
    return jsonify(response)

@app.route('/api/anomaly-scores')
def get_anomaly_scores():
    """
    API endpoint to provide REAL anomaly scores from the VAE model.
    """
    global current_index
    if not is_loaded:
        return jsonify({"error": "App dependencies not loaded."}), 500

    # 1. Get the latest window of data
    window_start = current_index - TIME_STEPS
    window_end = current_index
    data_window = data_df.iloc[window_start:window_end]

    # 2. Scale and reshape the data for the model
    scaled_window = scaler.transform(data_window)
    input_sequence = np.array([scaled_window]) # Shape: (1, TIME_STEPS, n_features)

    # 3. Get the model's reconstruction
    reconstruction = vae_model.predict(input_sequence, verbose=0)

    # 4. Calculate the reconstruction error
    mae_loss = np.mean(np.abs(reconstruction - input_sequence))
    
    # 5. Normalize the score for the dashboard (0 to 1+)
    # A score of 1.0 means it has hit the anomaly threshold
    normalized_score = mae_loss / anomaly_threshold
    
    return jsonify({
        'vae': min(1.5, normalized_score), # Cap at 1.5 to keep the gauge readable
        'arima': 0, # Placeholder for now
        'kalman': 0, # Placeholder for now
    })

@app.route('/api/xai-analysis')
def get_xai_analysis():
    """
    API endpoint for the XAI module using REAL model data.
    """
    if not is_loaded:
        return jsonify({"error": "App dependencies not loaded."}), 500

    # Get the original data point (the most recent one in the sequence)
    original_data = data_df.iloc[current_index - 1].values

    # Get the model's reconstruction for that same time step
    window_start = current_index - TIME_STEPS
    window_end = current_index
    input_sequence = np.array([scaler.transform(data_df.iloc[window_start:window_end])])
    
    reconstruction_sequence = vae_model.predict(input_sequence, verbose=0)[0]
    
    # Inverse transform to get back to original sensor values
    reconstructed_data_original_scale = scaler.inverse_transform(reconstruction_sequence)[-1]

    analysis = analyze_reconstruction_error(original_data, reconstructed_data_original_scale, SENSOR_NAMES)
    
    return jsonify(analysis)


if __name__ == '__main__':
    if is_loaded:
        app.run(debug=True, port=5001)
    else:
        print("Application failed to start due to loading errors.")
