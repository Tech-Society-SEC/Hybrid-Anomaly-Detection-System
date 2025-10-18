from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from xai_module import analyze_reconstruction_error # From xai_module.py
import os
import pickle

# --- Configuration ---
TIME_STEPS = 10                  # VAE sequence length (must match vae_model.py)
TRAIN_SPLIT_PERCENT = 0.8        # Must match training scripts
FILENAME = 'synthetic_sensor_data.csv'
VAE_MODEL_PATH = 'vae_model.h5'
KF_MODEL_PATH = 'kf_model.pkl'
ARIMA_CONFIG_PATH = 'arima_config.pkl'
BASELINE_TARGET_SENSOR = 'Sensor_4' # Sensor for ARIMA/Kalman (must match baseline_models.py)

app = Flask(__name__)

# --- Global Variables ---
vae_model, kf_model, arima_config = None, None, None
data_df, scaler = None, None
vae_anomaly_threshold = 0.0
SENSOR_NAMES = []
history = {} # To store rolling data for ARIMA

def load_dependencies():
    """Loads all models, the dataset, scaler, and calculates thresholds."""
    global vae_model, kf_model, arima_config, data_df, scaler, vae_anomaly_threshold, SENSOR_NAMES, history
    
    print("Loading dependencies for Flask app...")
    
    # 1. Load Data
    if not os.path.exists(FILENAME): return False, f"Dataset not found ({FILENAME}). Run eda.py."
    data_df = pd.read_csv(FILENAME, index_col='Timestamp', parse_dates=True)
    SENSOR_NAMES = [col for col in data_df.columns]
    # Pre-load history for ARIMA
    history[BASELINE_TARGET_SENSOR] = list(data_df[BASELINE_TARGET_SENSOR].iloc[:TIME_STEPS])

    # 2. Load All Three Models
    if not os.path.exists(VAE_MODEL_PATH): return False, f"VAE model not found ({VAE_MODEL_PATH}). Run vae_model.py."
    vae_model = tf.keras.models.load_model(VAE_MODEL_PATH, compile=False)
    
    if not os.path.exists(KF_MODEL_PATH): return False, f"Kalman Filter model not found ({KF_MODEL_PATH}). Run baseline_models.py."
    with open(KF_MODEL_PATH, 'rb') as f: kf_model = pickle.load(f)
        
    if not os.path.exists(ARIMA_CONFIG_PATH): return False, f"ARIMA config not found ({ARIMA_CONFIG_PATH}). Run baseline_models.py."
    with open(ARIMA_CONFIG_PATH, 'rb') as f: arima_config = pickle.load(f)

    # 3. Prepare Scaler and VAE Threshold (based on training data)
    train_size = int(len(data_df) * TRAIN_SPLIT_PERCENT)
    train_df = data_df.iloc[:train_size]
    scaler = MinMaxScaler().fit(train_df)
    
    # Calculate VAE threshold
    train_scaled = scaler.transform(train_df)
    X_train_seq = np.array([train_scaled[i:i+TIME_STEPS] for i in range(len(train_scaled) - TIME_STEPS)])
    X_train_pred = vae_model.predict(X_train_seq, verbose=0)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train_seq), axis=(1, 2))
    vae_anomaly_threshold = np.max(train_mae_loss) * 1.1 # 10% buffer

    print("--- All dependencies loaded successfully ---")
    print(f"VAE Anomaly Threshold: {vae_anomaly_threshold:.4f}")
    print(f"ARIMA Anomaly Threshold: {arima_config['threshold']:.4f}")
    print("------------------------------------------")
    return True, "Success"

# --- App Initialization ---
is_loaded, message = load_dependencies()
current_index = TIME_STEPS # Start after the first sequence window

# --- API Endpoints ---

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html', sensor_names=SENSOR_NAMES)

@app.route('/api/sensor-data')
def get_sensor_data():
    """Streams the next row of sensor data."""
    global current_index
    if not is_loaded: return jsonify({"error": message}), 500
    if current_index >= len(data_df): current_index = TIME_STEPS # Loop back
    
    data_point = data_df.iloc[current_index]
    response = {'timestamp': data_point.name.strftime('%Y-%m-%d %H:%M:%S'), 'sensors': data_point.to_dict()}
    current_index += 1
    return jsonify(response)

@app.route('/api/anomaly-scores')
def get_anomaly_scores():
    """Provides anomaly scores from all three models."""
    if not is_loaded: return jsonify({"error": message}), 500

    # --- VAE Score (Multivariate) ---
    window = data_df.iloc[current_index - TIME_STEPS : current_index]
    scaled_window = scaler.transform(window)
    input_seq = np.array([scaled_window])
    reconstruction = vae_model.predict(input_seq, verbose=0)
    mae_loss = np.mean(np.abs(reconstruction - input_seq))
    vae_score = mae_loss / vae_anomaly_threshold # Normalize score

    # --- Kalman Filter Score (Univariate on Target Sensor) ---
    target_series = data_df[BASELINE_TARGET_SENSOR].iloc[:current_index]
    # Filter the entire series up to the current point
    (smoothed_means, _) = kf_model.filter(target_series.values)
    # The error is the diff between the last point and its smoothed version
    kf_error = abs(target_series.values[-1] - smoothed_means[-1])
    # Normalize by the standard deviation of the training data (a simple z-score)
    kf_score = kf_error / (np.std(target_series.iloc[:int(len(data_df) * TRAIN_SPLIT_PERCENT)].values) + 1e-6)

    # --- ARIMA Score (Univariate on Target Sensor) ---
    target_val = data_df[BASELINE_TARGET_SENSOR].iloc[current_index - 1]
    history[BASELINE_TARGET_SENSOR].append(target_val)
    # Use a rolling window of 100 points for stable prediction
    model = ARIMA(history[BASELINE_TARGET_SENSOR][-100:], order=arima_config['order'])
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]
    arima_error = abs(target_val - prediction)
    arima_score = arima_error / arima_config['threshold'] # Normalize by pre-calculated threshold
    
    return jsonify({
        'vae': min(1.5, vae_score),   # Cap at 1.5 for readability on gauges
        'kalman': min(1.5, kf_score),
        'arima': min(1.5, arima_score)
    })

@app.route('/api/xai-analysis')
def get_xai_analysis():
    """Provides VAE-based root cause analysis."""
    if not is_loaded: return jsonify({"error": message}), 500

    # Get the original sensor values for the *last* time step
    original_data = data_df.iloc[current_index - 1].values
    
    # Get the VAE's reconstruction for that same time step
    window = data_df.iloc[current_index - TIME_STEPS : current_index]
    input_seq = np.array([scaler.transform(window)])
    recon_seq = vae_model.predict(input_seq, verbose=0)[0]
    
    # We only care about the last item in the reconstructed sequence
    # Inverse transform to get back to original sensor values for comparison
    reconstructed_data = scaler.inverse_transform(recon_seq)[-1] 

    analysis = analyze_reconstruction_error(original_data, reconstructed_data, SENSOR_NAMES)
    return jsonify(analysis)

if __name__ == '__main__':
    if is_loaded:
        app.run(debug=True, port=5001)
    else:
        print(f"Application failed to start: {message}")
