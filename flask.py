from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import random
from xai_module import analyze_reconstruction_error

app = Flask(__name__)

# --- Data Loading ---
try:
    df = pd.read_csv('synthetic_sensor_data.csv', index_col='Timestamp', parse_dates=True)
    SENSOR_NAMES = [col for col in df.columns]
    print("Successfully loaded synthetic_sensor_data.csv")
except FileNotFoundError:
    print("\nERROR: synthetic_sensor_data.csv not found.")
    print("Please run eda.py first to generate the data file.\n")
    df = None # Set to None to handle error gracefully
    SENSOR_NAMES = [f'Sensor_{i+1}' for i in range(5)]


# Global variable to track the current position in the data stream
current_index = 0

# --- API Endpoints ---

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html', sensor_names=SENSOR_NAMES)

@app.route('/api/sensor-data')
def get_sensor_data():
    """API endpoint to stream the next row of sensor data."""
    global current_index
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    if current_index >= len(df):
        current_index = 0  # Loop back to the start

    data_point = df.iloc[current_index]
    
    response = {
        'timestamp': data_point.name.strftime('%Y-%m-%d %H:%M:%S'),
        'sensors': {name: val for name, val in zip(SENSOR_NAMES, data_point.values)}
    }
    
    current_index += 1
    return jsonify(response)

@app.route('/api/anomaly-scores')
def get_anomaly_scores():
    """
    API endpoint to provide anomaly scores.
    **THIS IS A PLACEHOLDER**. In Phase 2, this will be replaced with real model outputs.
    """
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500
        
    # Simulate a VAE score that gets high during the known anomaly period
    is_in_anomaly_window = (current_index > (len(df) * 0.85)) and (current_index < (len(df) * 0.90))
    
    if is_in_anomaly_window:
        vae_score = random.uniform(0.8, 1.0)
    else:
        vae_score = random.uniform(0.05, 0.2)
        
    # Simulate other scores
    arima_score = vae_score * random.uniform(0.7, 1.1)
    kalman_score = vae_score * random.uniform(0.8, 1.2)

    return jsonify({
        'vae': min(1.0, vae_score), # Cap at 1.0
        'arima': min(1.0, arima_score),
        'kalman': min(1.0, kalman_score),
    })

@app.route('/api/xai-analysis')
def get_xai_analysis():
    """
    API endpoint for the XAI module.
    **THIS IS A PLACEHOLDER**. It simulates a VAE's reconstruction error.
    """
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    # Get the original data point
    original_data = df.iloc[current_index -1].values

    # Simulate a reconstructed data point
    # During an anomaly, the reconstruction will be poor for specific sensors
    reconstructed_data = original_data.copy()
    is_in_anomaly_window = (current_index > (len(df) * 0.85)) and (current_index < (len(df) * 0.90))
    
    if is_in_anomaly_window:
        # Simulate poor reconstruction for Sensor 2 and 4 (0-indexed)
        reconstructed_data[1] = original_data[1] * 0.6 + np.random.normal(0, 5)
        reconstructed_data[3] = original_data[3] * 0.7 + np.random.normal(0, 5)
    else:
        # Simulate a good reconstruction
        reconstructed_data += np.random.normal(0, 0.5, len(SENSOR_NAMES))

    analysis = analyze_reconstruction_error(original_data, reconstructed_data, SENSOR_NAMES)
    
    return jsonify(analysis)


if __name__ == '__main__':
    # Using a higher port to avoid conflicts with common ports like 5000
    app.run(debug=True, port=5001)


