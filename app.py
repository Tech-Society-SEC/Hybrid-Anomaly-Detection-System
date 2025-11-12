"""
Flask Backend Server for Predictive Maintenance Dashboard
Run this on your local machine after downloading trained models from Kaggle
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import os
import warnings
warnings.filterwarnings('ignore')

@keras.utils.register_keras_serializable()
class Sampling(keras.layers.Layer):
    """Reparameterization trick layer used in the VAE encoder."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__, static_folder='frontend')
CORS(app)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Models and data will be loaded on startup
models = {}
test_data = None
scaler = None
feature_info = None
current_index = 0
kalman_state = None

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_models():
    """Load all trained models"""
    global models, scaler, feature_info
    
    print("Loading models...")
    
    # Load scaler
    with open('trained_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
    
    # Load feature info
    with open('trained_models/feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    print("✓ Feature info loaded")
    
    # Load ARIMA model
    with open('trained_models/arima_model.pkl', 'rb') as f:
        models['arima'] = pickle.load(f)
    print("✓ ARIMA model loaded")
    
    # Load Kalman Filter model
    with open('trained_models/kalman_model.pkl', 'rb') as f:
        models['kalman'] = pickle.load(f)
    print("✓ Kalman Filter loaded")
    
    # Load VAE model
    vae_model_paths = [
        'trained_models/vae_model.h5',
        'trained_models/vae_model.keras'
    ]
    vae_model_path = next((path for path in vae_model_paths if os.path.exists(path)), None)
    models['vae'] = None

    custom_objects = {'Sampling': Sampling}

    if vae_model_path is not None:
        try:
            models['vae'] = keras.models.load_model(
                vae_model_path,
                compile=False,
                custom_objects=custom_objects
            )
            print(f"✓ VAE model loaded ({vae_model_path})")
        except TypeError as e:
            if "class 'VAE'" in str(e):
                print("ℹ Detected custom VAE class serialization; falling back to encoder/decoder weights.")
            else:
                raise
    else:
        print("ℹ VAE model file not found in expected locations. Trying encoder/decoder weights.")

    if models['vae'] is None:
        encoder_path = 'trained_models/vae_encoder.keras'
        decoder_path = 'trained_models/vae_decoder.keras'
        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            raise FileNotFoundError(
                "Unable to load VAE model. Provide either one of the following files: "
                + ", ".join(vae_model_paths)
                + " or both encoder/decoder files: "
                + f"{encoder_path}, {decoder_path}"
            )
        models['vae_encoder'] = keras.models.load_model(
            encoder_path,
            compile=False,
            custom_objects=custom_objects
        )
        models['vae_decoder'] = keras.models.load_model(
            decoder_path,
            compile=False,
            custom_objects=custom_objects
        )
        print(f"✓ VAE encoder loaded ({encoder_path})")
        print(f"✓ VAE decoder loaded ({decoder_path})")
    
    # Load VAE config
    with open('trained_models/vae_config.pkl', 'rb') as f:
        models['vae_config'] = pickle.load(f)
    print("✓ VAE config loaded")
    
    print("\nAll models loaded successfully!")

def load_test_data():
    """Load and preprocess test data"""
    global test_data
    
    print("\nLoading test data...")
    
    # Load processed test data if available
    if os.path.exists('data/processed_test_data.pkl'):
        test_data = pd.read_pickle('data/processed_test_data.pkl')
        print("✓ Loaded preprocessed test data")
    else:
        # Load raw test data and process it
        test_data = pd.read_pickle('trained_models/test_scaled.pkl')
        print("✓ Loaded test data")
    
    print(f"Test data shape: {test_data.shape}")

# ============================================================================
# ANOMALY DETECTION FUNCTIONS
# ============================================================================

def get_arima_score(sensor_value):
    """Calculate ARIMA anomaly score"""
    try:
        arima_config = models['arima']
        arima_model = arima_config['model']
        
        # Simple forecast-based anomaly score
        forecast = arima_model.forecast(steps=1)[0]
        score = float(abs(sensor_value - forecast))
        
        return score
    except Exception as e:
        print(f"ARIMA error: {e}")
        return 0.0

def get_kalman_score(sensor_values):
    """Calculate Kalman Filter anomaly score"""
    global kalman_state
    
    try:
        kalman_config = models['kalman']
        kf = kalman_config['model']
        key_sensors = kalman_config['key_sensors']
        
        # Initialize state if needed
        if kalman_state is None:
            # Initialize with first observation
            kalman_state = {
                'mean': np.zeros(kf.n_dim_state),
                'covariance': kf.initial_state_covariance
            }
        
        observation = np.array([sensor_values.get(s, 0.0) for s in key_sensors])
        
        # Predict
        predicted_mean = kf.transition_matrices @ kalman_state['mean']
        predicted_cov = (
            kf.transition_matrices @ 
            kalman_state['covariance'] @ 
            kf.transition_matrices.T + 
            kf.transition_covariance
        )
        
        # Calculate innovation
        predicted_obs = kf.observation_matrices @ predicted_mean
        innovation = observation - predicted_obs
        
        # Innovation covariance
        innovation_cov = (
            kf.observation_matrices @ 
            predicted_cov @ 
            kf.observation_matrices.T + 
            kf.observation_covariance
        )
        
        # Mahalanobis distance
        inv_cov = np.linalg.inv(innovation_cov + np.eye(innovation_cov.shape[0]) * 1e-6)
        mahalanobis = np.sqrt(innovation.T @ inv_cov @ innovation)
        
        # Update state
        kalman_state['mean'], kalman_state['covariance'] = kf.filter_update(
            predicted_mean,
            predicted_cov,
            observation
        )
        
        return float(mahalanobis)
        
    except Exception as e:
        print(f"Kalman error: {e}")
        return 0.0

def get_vae_score_and_explanation(window_data):
    """Calculate VAE anomaly score and explanation"""
    try:
        vae_model = models['vae']
        vae_config = models['vae_config']
        sensor_cols = vae_config['sensor_cols']
        
        # Prepare window
        if len(window_data.shape) == 2:
            window_data = np.expand_dims(window_data, axis=0)
        
        # Get reconstruction
        if vae_model is not None:
            reconstructed = vae_model.predict(window_data, verbose=0)
        else:
            vae_encoder = models['vae_encoder']
            vae_decoder = models['vae_decoder']
            latent_outputs = vae_encoder.predict(window_data, verbose=0)
            if isinstance(latent_outputs, (list, tuple)):
                z = latent_outputs[-1]
            else:
                z = latent_outputs
            reconstructed = vae_decoder.predict(z, verbose=0)
        
        # Calculate overall anomaly score (MSE)
        anomaly_score = float(np.mean(np.square(window_data - reconstructed)))
        
        # Calculate per-sensor explanation (MAE over time)
        per_sensor_error = np.mean(np.abs(window_data[0] - reconstructed[0]), axis=0)
        
        explanation = {
            sensor: float(error)
            for sensor, error in zip(sensor_cols, per_sensor_error)
        }
        
        return anomaly_score, explanation
        
    except Exception as e:
        print(f"VAE error: {e}")
        return 0.0, {}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the dashboard"""
    return send_from_directory('frontend', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('frontend/static', path)

@app.route('/api/get-live-data-and-scores', methods=['GET'])
def get_live_data_and_scores():
    """Main API endpoint - returns sensor data and anomaly scores"""
    global current_index, test_data
    
    try:
        # Check if we've reached the end
        if current_index >= len(test_data):
            current_index = 0  # Reset to beginning
        
        # Get current data point
        current_row = test_data.iloc[current_index]
        sensor_cols = feature_info['sensor_cols']
        window_size = feature_info['window_size']
        
        # Prepare sensor data
        sensor_data = {
            sensor: float(current_row[sensor])
            for sensor in sensor_cols
        }
        
        # Get ARIMA score (using key sensor)
        arima_key_sensor = models['arima']['key_sensor']
        arima_score = get_arima_score(sensor_data.get(arima_key_sensor, 0.0))
        
        # Get Kalman score
        kalman_score = get_kalman_score(sensor_data)
        
        # Get VAE score and explanation (need window)
        # For simplicity, create a window from recent data
        start_idx = max(0, current_index - window_size + 1)
        end_idx = current_index + 1
        
        if end_idx - start_idx == window_size:
            window_df = test_data.iloc[start_idx:end_idx][sensor_cols]
            window_array = window_df.values
            vae_score, vae_explanation = get_vae_score_and_explanation(window_array)
        else:
            # Not enough data for full window yet
            vae_score = 0.0
            vae_explanation = {}
        
        # Determine if anomaly detected
        vae_threshold = models['vae_config']['anomaly_threshold']
        is_anomaly = vae_score > vae_threshold
        
        # Prepare response
        response = {
            'cycle': int(current_row['cycle']),
            'unit': int(current_row['unit']),
            'live_sensor_data': sensor_data,
            'anomaly_scores': {
                'arima': round(arima_score, 6),
                'kalman': round(kalman_score, 6),
                'vae': round(vae_score, 6)
            },
            'thresholds': {
                'vae': round(vae_threshold, 6),
                'kalman': round(models['kalman']['anomaly_threshold'], 6)
            },
            'is_anomaly': is_anomaly,
            'xai_explanation': vae_explanation if is_anomaly else None
        }
        
        # Increment index for next call
        current_index += 1
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in get_live_data_and_scores: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset to beginning of test data"""
    global current_index, kalman_state
    current_index = 0
    kalman_state = None
    return jsonify({'status': 'reset', 'message': 'Data stream reset to beginning'})

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        'status': 'running',
        'models_loaded': len(models) > 0,
        'current_index': current_index,
        'total_datapoints': len(test_data) if test_data is not None else 0
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("PREDICTIVE MAINTENANCE DASHBOARD - BACKEND SERVER")
    print("="*60)
    
    # Load models and data
    load_models()
    load_test_data()
    
    print("\n" + "="*60)
    print("SERVER STARTING...")
    print("="*60)
    print("\nDashboard URL: http://localhost:5000")
    print("API Status: http://localhost:5000/api/status")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)