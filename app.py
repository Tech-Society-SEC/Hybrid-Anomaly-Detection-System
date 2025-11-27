# Revised app.py with SHAP integration
"""
Flask API Backend for Hybrid Time Series Anomaly Detection
Real-time monitoring and explainability system - ENHANCED FOR XAI with SHAP
"""

from flask import Flask, jsonify, request, render_template, Response
import os
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import threading
import time
from datetime import datetime
from collections import deque
import json
import shap  # Added for SHAP explainability

app = Flask(__name__)
CORS(app)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class SystemState:
    def __init__(self):
        self.models_loaded = False
        self.simulation_running = False
        self.current_engine = 1
        self.current_cycle = 0
        self.sensor_buffer = deque(maxlen=30)  # Window size
        self.anomaly_history = deque(maxlen=100)
        self.sensor_data_history = deque(maxlen=100)
        
        # Per-Engine History Storage for detailed analytics
        self.engine_history = {}  # {engine_id: {'anomalies': [], 'cycles': [], 'health_score': float, 'total_cycles': int}}
        self.completed_engines = []  # List of engine_ids that completed all cycles

        # XAI Data Storage - Enhanced with SHAP
        self.last_window_actual = None
        self.last_window_recon = None
        self.last_sensor_errors = {}
        self.last_kalman_innovations = {}  # Per-sensor innovation for Kalman
        self.last_arima_residuals = {}     # Per-sensor residuals for ARIMA
        self.last_shap_values = None       # SHAP values for VAE
        
        # Models
        self.encoder = None
        self.decoder = None
        self.vae_model = None  # Combined VAE for SHAP
        self.kalman_model = None
        self.arima_models = {}
        self.shap_explainer = None  # SHAP explainer
        
        # Configs
        self.vae_config = {}
        self.kalman_config = {}
        self.arima_config = {}
        self.feature_info = {}
        
        # State variables
        self.kalman_state_mean = None
        self.kalman_state_cov = None
        self.shap_background = None  # Background data for SHAP
        
state = SystemState()

# ============================================================================
# CUSTOM KERAS LAYERS (Must match training exactly)
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="CustomModels", name="Sampling")
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.utils.register_keras_serializable(package="CustomModels", name="AttentionLayer")
class AttentionLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = keras.layers.Dense(units)
        self.U = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)
    
    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ============================================================================
# MODEL LOADING - Enhanced with SHAP
# ============================================================================

def load_models():
    """Load all trained models and configurations, including SHAP explainer.
    Adds robust fallbacks so demo runs even if some pickle files are missing."""
    try:
        print("Loading models...")
        custom_objects = {'Sampling': Sampling, 'AttentionLayer': AttentionLayer}

        # 1. Load config first (original code attempted after using it)
        vae_config = {}
        if os.path.exists('vae_config.pkl'):
            try:
                with open('vae_config.pkl', 'rb') as f:
                    vae_config = pickle.load(f)
            except Exception as e:
                print(f"Warning: could not read vae_config.pkl ({e})")
        state.vae_config = vae_config

        # 2. Load encoder/decoder
        state.encoder = keras.models.load_model('vae_encoder.keras', compile=False, custom_objects=custom_objects)
        state.decoder = keras.models.load_model('vae_decoder.keras', compile=False, custom_objects=custom_objects)

        # 3. Infer window_size and sensor_cols if missing in config
        inferred_window = None
        inferred_sensors = None
        try:
            enc_shape = state.encoder.input_shape  # (None, window, features)
            if isinstance(enc_shape, list):
                enc_shape = enc_shape[0]
            if len(enc_shape) == 3:
                inferred_window = enc_shape[1]
                inferred_feat = enc_shape[2]
                inferred_sensors = [f"sensor_{i}" for i in range(inferred_feat)]
        except Exception as e:
            print(f"Warning: could not infer encoder input shape ({e})")

        window_size = state.vae_config.get('window_size') or state.vae_config.get('seq_len') or inferred_window
        sensor_cols = state.vae_config.get('sensor_cols') or state.vae_config.get('features') or inferred_sensors
        if window_size is None or sensor_cols is None:
            raise ValueError("Missing window_size or sensor_cols (unable to infer). Provide vae_config.pkl or retrain models.")
        state.vae_config['window_size'] = window_size
        state.vae_config['sensor_cols'] = sensor_cols

        # 4. Build combined VAE for reconstruction
        input_layer = keras.Input(shape=(window_size, len(sensor_cols)))
        encoded = state.encoder(input_layer)
        z = encoded[2] if isinstance(encoded, list) and len(encoded) >= 3 else (encoded[-1] if isinstance(encoded, list) else encoded)
        reconstructed = state.decoder(z)
        vae_full = keras.Model(input_layer, reconstructed)
        vae_full.compile(optimizer='adam', loss='mse')
        state.vae_model = vae_full

        # 5. SHAP background: prefer shap_background.pkl, else normal_windows.npy, else synthetic
        background = None
        if os.path.exists('shap_background.pkl'):
            try:
                with open('shap_background.pkl', 'rb') as f:
                    background = pickle.load(f)
                print("Loaded shap_background.pkl")
            except Exception as e:
                print(f"Warning: shap_background.pkl unreadable ({e})")
        if background is None and os.path.exists('normal_windows.npy'):
            try:
                arr = np.load('normal_windows.npy')
                # Ensure shape (N, window, features)
                if arr.ndim == 3 and arr.shape[1] == window_size:
                    background = arr[:min(100, arr.shape[0])]
                    print("Using normal_windows.npy as SHAP background")
            except Exception as e:
                print(f"Warning: normal_windows.npy unusable ({e})")
        if background is None:
            background = np.random.normal(0.0, 1.0, (64, window_size, len(sensor_cols)))
            print("Using synthetic SHAP background (fallback)")
        state.shap_background = background

        try:
            state.shap_explainer = shap.DeepExplainer(state.vae_model, state.shap_background)
        except Exception as e:
            state.shap_explainer = None
            print(f"Warning: SHAP DeepExplainer init failed ({e}). Explainability will degrade.")

        # 6. Kalman config
        if os.path.exists('kalman_model.pkl'):
            try:
                with open('kalman_model.pkl', 'rb') as f:
                    state.kalman_config = pickle.load(f)
                state.kalman_model = state.kalman_config.get('model')
            except Exception as e:
                print(f"Warning: kalman_model.pkl load failed ({e})")
        else:
            state.kalman_config = {'key_sensors': sensor_cols[:min(5, len(sensor_cols))], 'threshold': 12.0}
            state.kalman_model = None
            print("Kalman model missing – using placeholder config.")

        # 7. ARIMA config
        if os.path.exists('arima_model.pkl'):
            try:
                with open('arima_model.pkl', 'rb') as f:
                    state.arima_config = pickle.load(f)
                state.arima_models = state.arima_config.get('models', {})
            except Exception as e:
                print(f"Warning: arima_model.pkl load failed ({e})")
        else:
            state.arima_config = {'key_sensors': sensor_cols[:min(5, len(sensor_cols))], 'threshold': 0.065}
            state.arima_models = {}
            print("ARIMA models missing – using placeholder config.")

        # 8. Feature info (optional)
        if os.path.exists('feature_info.pkl'):
            try:
                with open('feature_info.pkl', 'rb') as f:
                    state.feature_info = pickle.load(f)
            except Exception as e:
                print(f"Warning: feature_info.pkl load failed ({e})")
        else:
            state.feature_info = {s: {'desc': s} for s in sensor_cols}

        # 9. Test data for simulation
        if os.path.exists('test_scaled.pkl'):
            try:
                state.test_data = pd.read_pickle('test_scaled.pkl')
            except Exception as e:
                print(f"Warning: test_scaled.pkl unreadable ({e}); simulation disabled until provided.")
                state.test_data = pd.DataFrame()
        else:
            print("Warning: test_scaled.pkl missing; simulation will not advance.")
            state.test_data = pd.DataFrame()

        state.models_loaded = True
        print("✓ Models loaded (with fallbacks). window_size=%d sensors=%d" % (window_size, len(sensor_cols)))
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

# ============================================================================
# ANOMALY DETECTION FUNCTIONS - Enhanced with SHAP
# ============================================================================

def compute_vae_score(window):
    """Compute VAE reconstruction error, SHAP values, and store XAI data"""
    try:
        window_batch = np.expand_dims(window, axis=0).astype(np.float32)
        reconstructed = state.vae_model.predict(window_batch, verbose=0)[0]
        
        # Store for XAI
        state.last_window_actual = window[-1]  # Last timestep
        state.last_window_recon = reconstructed[-1]  # Last timestep recon
        
        error = float(np.mean(np.square(window - reconstructed)))
        
        # Per-sensor errors
        sensor_errors = {}
        for i, sensor in enumerate(state.vae_config['sensor_cols']):
            sensor_error = float(np.mean(np.square(window[:, i] - reconstructed[:, i])))
            sensor_errors[sensor] = sensor_error
        state.last_sensor_errors = sensor_errors
        
        # Compute SHAP values (only if explainer initialized)
        shap_per_sensor = {}
        if state.shap_explainer is not None:
            try:
                shap_values = state.shap_explainer.shap_values(window_batch)[0]  # For reconstruction
                # Aggregate SHAP per sensor (mean over time steps)
                n_time, n_sens = window.shape
                for i, sensor in enumerate(state.vae_config['sensor_cols']):
                    shap_per_sensor[sensor] = float(np.mean(shap_values[:, :n_time*n_sens:n_sens + i]))  # Simplified indexing
            except Exception as shap_err:
                # SHAP computation failed, continue without it
                shap_per_sensor = {}
        state.last_shap_values = shap_per_sensor
        
        return error, sensor_errors, reconstructed, shap_per_sensor
    except Exception as e:
        print(f"VAE/SHAP error: {e}")
        return 0.0, {}, None, {}

def compute_kalman_score(observation):
    """Enhanced Kalman with per-sensor innovation breakdown"""
    try:
        if state.kalman_state_mean is None:
            state.kalman_state_mean = np.zeros(len(state.kalman_config['key_sensors']))
            state.kalman_state_cov = np.eye(len(state.kalman_config['key_sensors']))
        
        kf = state.kalman_model
        predicted_obs = kf.observation_matrices @ state.kalman_state_mean
        innovation = observation - predicted_obs
        
        # Store per-sensor innovations for XAI
        sensor_innovations = {}
        for i, sensor in enumerate(state.kalman_config['key_sensors']):
            sensor_innovations[sensor] = float(abs(innovation[i]))
        state.last_kalman_innovations = sensor_innovations
        
        S = (kf.observation_matrices @ state.kalman_state_cov @ kf.observation_matrices.T + 
             kf.observation_covariance)
        S_reg = S + np.eye(S.shape[0]) * 1e-8
        
        try:
            S_inv = np.linalg.inv(S_reg)
        except:
            eigvals, eigvecs = np.linalg.eigh(S_reg)
            eigvals[eigvals < 1e-8] = 1e-8
            S_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        
        mahal_dist = float(innovation.T @ S_inv @ innovation)
        
        state.kalman_state_mean, state.kalman_state_cov = kf.filter_update(
            state.kalman_state_mean, state.kalman_state_cov, observation
        )
        return mahal_dist
    except Exception as e:
        print(f"Kalman error: {e}")
        return 0.0

def compute_arima_score(sensor_data):
    """Enhanced ARIMA with per-sensor residuals"""
    try:
        total_error = 0.0
        count = 0
        residuals = {}
        for sensor in state.arima_config['key_sensors']:
            if sensor in sensor_data.columns and len(sensor_data) >= 10:
                history = sensor_data[sensor].values[-10:]
                forecast = np.mean(history)  # Simple mean; replace with ARIMA forecast
                actual = sensor_data[sensor].values[-1]
                residual = abs(actual - forecast)
                residuals[sensor] = float(residual)
                total_error += residual
                count += 1
        state.last_arima_residuals = residuals
        return total_error / count if count > 0 else 0.0
    except Exception as e:
        print(f"ARIMA error: {e}")
        return 0.0

# ============================================================================
# SIMULATION ENGINE - Updated to capture SHAP
# ============================================================================

def simulation_loop():
    while state.simulation_running:
        try:
            engine_data = state.test_data[state.test_data['unit'] == state.current_engine]
            
            if state.current_cycle >= len(engine_data):
                state.current_engine += 1
                if state.current_engine > state.test_data['unit'].max():
                    state.current_engine = 1
                state.current_cycle = 0
                state.sensor_buffer.clear()
                state.kalman_state_mean = None
                continue
            
            current_row = engine_data.iloc[state.current_cycle]
            sensor_cols = state.vae_config['sensor_cols']
            sensor_readings = {col: float(current_row[col]) for col in sensor_cols}
            sensor_readings['cycle'] = int(current_row['cycle'])
            sensor_readings['timestamp'] = datetime.now().isoformat()
            
            state.sensor_buffer.append(np.array([sensor_readings[col] for col in sensor_cols]))
            state.sensor_data_history.append(sensor_readings)
            
            anomaly_data = {
                'cycle': int(current_row['cycle']),
                'vae_score': 0.0,
                'kalman_score': 0.0,
                'arima_score': 0.0,
                'is_anomaly': False,
                'confidence': 0.0
            }
            
            if len(state.sensor_buffer) == 30:
                # VAE with SHAP
                window = np.array(list(state.sensor_buffer))
                vae_score, _, _, shap_vals = compute_vae_score(window)
                anomaly_data['vae_score'] = vae_score
                
                # Kalman
                kalman_obs = np.array([sensor_readings[s] for s in state.kalman_config['key_sensors']])
                kalman_score = compute_kalman_score(kalman_obs)
                anomaly_data['kalman_score'] = kalman_score
                
                # ARIMA
                buffer_df = pd.DataFrame(list(state.sensor_buffer), columns=sensor_cols)
                arima_score = compute_arima_score(buffer_df)
                anomaly_data['arima_score'] = arima_score
                
                # Ensemble
                vae_threshold = state.vae_config.get('anomaly_threshold', 0.005)
                kalman_threshold = state.kalman_config.get('threshold', 12.0)
                arima_threshold = state.arima_config.get('threshold', 0.065)
                
                vae_binary = 1 if vae_score > vae_threshold else 0
                kalman_binary = 1 if kalman_score > kalman_threshold else 0
                arima_binary = 1 if arima_score > arima_threshold else 0
                
                weights = {'vae': 0.4, 'kalman': 0.3, 'arima': 0.3}
                weighted_vote = (weights['vae'] * vae_binary + 
                               weights['kalman'] * kalman_binary + 
                               weights['arima'] * arima_binary)
                
                anomaly_data['is_anomaly'] = weighted_vote > 0.45
                anomaly_data['confidence'] = float(weighted_vote)
            
            state.anomaly_history.append(anomaly_data)
            
            # Track per-engine history for detailed analytics
            if state.current_engine not in state.engine_history:
                state.engine_history[state.current_engine] = {
                    'anomalies': [],
                    'cycles': [],
                    'health_score': 100.0,
                    'total_cycles': 0,
                    'sensors_snapshot': {}
                }
            
            engine_record = state.engine_history[state.current_engine]
            engine_record['total_cycles'] = state.current_cycle
            
            # Store sensor snapshot for every cycle (for sensor contribution analysis)
            engine_record['sensors_snapshot'] = sensor_readings.copy()
            
            if anomaly_data['is_anomaly']:
                anomaly_record = {
                    'cycle': anomaly_data['cycle'],
                    'vae_score': anomaly_data['vae_score'],
                    'confidence': anomaly_data['confidence'],
                    'sensors': sensor_readings.copy(),
                    'timestamp': sensor_readings['timestamp'],
                    'explanation': build_explainability_payload() if len(state.sensor_buffer) == 30 else None
                }
                engine_record['anomalies'].append(anomaly_record)
                print(f"[ANOMALY] Engine {state.current_engine}, Cycle {anomaly_data['cycle']}, Confidence: {anomaly_data['confidence']:.2f}, Total anomalies: {len(engine_record['anomalies'])}")
                # Degrade health score
                engine_record['health_score'] = max(0, engine_record['health_score'] - (anomaly_data['confidence'] * 2))
            
            # Track cycle with full sensor data and model scores
            cycle_data = {
                'cycle': int(current_row['cycle']),
                'vae_score': anomaly_data['vae_score'],
                'kalman_score': anomaly_data['kalman_score'],
                'arima_score': anomaly_data['arima_score'],
                'confidence': anomaly_data['confidence']
            }
            # Add sensor errors if available (after window fills)
            if len(state.sensor_buffer) == 30 and state.last_sensor_errors:
                cycle_data['sensor_errors'] = state.last_sensor_errors.copy()
            if state.last_shap_values:
                cycle_data['shap_values'] = state.last_shap_values.copy()
            
            engine_record['cycles'].append(cycle_data)
            
            # Check if engine completed all cycles (mark as completed)
            if len(engine_data) > 0:
                max_cycle_for_engine = len(engine_data)
                if state.current_cycle >= max_cycle_for_engine - 1:
                    if state.current_engine not in state.completed_engines:
                        state.completed_engines.append(state.current_engine)
                        print(f"[INFO] Engine {state.current_engine} completed all {max_cycle_for_engine} cycles - moved to archive")
            
            state.current_cycle += 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Simulation error: {e}")
            time.sleep(1)

# ============================================================================
# API ENDPOINTS - Enhanced with SHAP
# ============================================================================

@app.route('/engine/<int:engine_id>')
def engine_detail(engine_id):
    return render_template('engine_detail.html', engine_id=engine_id)

@app.route('/api/engine/<int:engine_id>/analysis')
def get_engine_analysis(engine_id):
    if engine_id not in state.engine_history:
        return jsonify({'error': 'Engine not found or not yet processed'}), 404
    
    record = state.engine_history[engine_id]
    
    # Calculate statistics
    total_anomalies = len(record['anomalies'])
    anomaly_rate = (total_anomalies / max(record['total_cycles'], 1)) * 100
    avg_vae = sum(c['vae_score'] for c in record['cycles']) / max(len(record['cycles']), 1) if record['cycles'] else 0
    
    # Get sensor statistics from ALL CYCLES (not just anomalies) for robust analysis
    sensor_contributions = {}
    
    # Method 1: Use sensor_errors and shap_values from cycle data (available after window fills)
    for cycle in record['cycles']:
        if 'sensor_errors' in cycle:
            for sensor, error in cycle['sensor_errors'].items():
                if sensor not in sensor_contributions:
                    sensor_contributions[sensor] = {'count': 0, 'total_error': 0.0, 'total_shap': 0.0}
                sensor_contributions[sensor]['count'] += 1
                sensor_contributions[sensor]['total_error'] += error
        
        if 'shap_values' in cycle:
            for sensor, shap in cycle['shap_values'].items():
                if sensor not in sensor_contributions:
                    sensor_contributions[sensor] = {'count': 0, 'total_error': 0.0, 'total_shap': 0.0}
                if sensor_contributions[sensor]['count'] == 0:
                    sensor_contributions[sensor]['count'] = 1
                sensor_contributions[sensor]['total_shap'] += abs(shap)
    
    # Method 2: Fallback to anomaly explanations if cycle data unavailable
    if not sensor_contributions and record['anomalies']:
        for anom in record['anomalies']:
            if anom.get('explanation') and anom['explanation'].get('top_features'):
                for feat in anom['explanation']['top_features'][:5]:
                    sensor = feat['sensor']
                    if sensor not in sensor_contributions:
                        sensor_contributions[sensor] = {'count': 0, 'total_error': 0.0, 'total_shap': 0.0}
                    sensor_contributions[sensor]['count'] += 1
                    sensor_contributions[sensor]['total_error'] += feat['vae_error']
                    sensor_contributions[sensor]['total_shap'] += abs(feat['shap_value'])
    
    # Sort sensors by total contribution (weighted average of error and SHAP)
    top_sensors = []
    if sensor_contributions:
        for sensor, data in sensor_contributions.items():
            avg_error = data['total_error'] / data['count'] if data['count'] > 0 else 0
            avg_shap = data['total_shap'] / data['count'] if data['count'] > 0 else 0
            # Contribution score: higher error + higher SHAP = higher contribution
            contribution_score = avg_error + (avg_shap * 10)  # Weight SHAP higher
            top_sensors.append({
                'sensor': sensor,
                'count': data['count'],
                'avg_error': avg_error,
                'avg_shap': avg_shap,
                'contribution_score': contribution_score
            })
        top_sensors = sorted(top_sensors, key=lambda x: x['contribution_score'], reverse=True)[:10]
    
    return jsonify({
        'engine_id': engine_id,
        'total_cycles': record['total_cycles'],
        'total_anomalies': total_anomalies,
        'anomaly_rate': anomaly_rate,
        'health_score': record['health_score'],
        'avg_vae_score': avg_vae,
        'cycles': record['cycles'],
        'anomalies': record['anomalies'],
        'top_sensors': top_sensors,
        'status': 'CRITICAL' if record['health_score'] < 50 else 'WARNING' if record['health_score'] < 80 else 'HEALTHY'
    })

@app.route('/api/completed-engines')
def get_completed_engines():
    """Return list of engines that completed all cycles"""
    completed = []
    for engine_id in state.completed_engines:
        if engine_id in state.engine_history:
            record = state.engine_history[engine_id]
            completed.append({
                'engine_id': engine_id,
                'total_cycles': record['total_cycles'],
                'total_anomalies': len(record['anomalies']),
                'health_score': record['health_score'],
                'status': 'CRITICAL' if record['health_score'] < 50 else 'WARNING' if record['health_score'] < 80 else 'HEALTHY'
            })
    return jsonify({'completed_engines': completed})

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/start', methods=['POST'])
def start_simulation():
    if not state.models_loaded:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    if not state.simulation_running:
        state.simulation_running = True
        thread = threading.Thread(target=simulation_loop, daemon=True)
        thread.start()
    return jsonify({'message': 'Simulation started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    state.simulation_running = False
    return jsonify({'message': 'Simulation stopped'})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    data = request.get_json()
    state.current_engine = data.get('engine_id', 1)
    state.current_cycle = 0
    state.sensor_buffer.clear()
    state.anomaly_history.clear()
    state.sensor_data_history.clear()
    state.kalman_state_mean = None
    return jsonify({'message': 'Reset'})

@app.route('/api/current_data')
def get_current_data():
    if len(state.sensor_data_history) == 0:
        return jsonify({'error': 'No data'}), 404
    return jsonify({
        'sensors': list(state.sensor_data_history)[-1],
        'anomaly': list(state.anomaly_history)[-1] if len(state.anomaly_history) > 0 else None,
        'engine': state.current_engine,
        'cycle': state.current_cycle
    })

@app.route('/api/history')
def get_history():
    n = int(request.args.get('n', 50))
    return jsonify({
        'sensors': list(state.sensor_data_history)[-n:],
        'anomalies': list(state.anomaly_history)[-n:]
    })

def build_explainability_payload():
    if len(state.anomaly_history) == 0 or state.last_window_actual is None:
        return None
    current_anomaly = list(state.anomaly_history)[-1]
    vae_errors = state.last_sensor_errors
    if not vae_errors:
        return None
    mean_vae_error = np.mean(list(vae_errors.values()))
    max_vae_error = np.max(list(vae_errors.values()))
    vae_analysis = f"Mean recon err: {mean_vae_error:.4f}, Max: {max_vae_error:.4f}"
    shap_vals = state.last_shap_values or {}
    kalman_innov = state.last_kalman_innovations
    arima_res = state.last_arima_residuals
    all_contribs = {}
    for sensor, error in sorted(vae_errors.items(), key=lambda x: x[1], reverse=True)[:3]:
        all_contribs[sensor] = {'vae': error, 'shap': shap_vals.get(sensor, 0), 'kalman': kalman_innov.get(sensor, 0), 'arima': arima_res.get(sensor, 0), 'total': error + abs(shap_vals.get(sensor, 0))}
    for sensor, innov in sorted(kalman_innov.items(), key=lambda x: x[1], reverse=True)[:3]:
        if sensor not in all_contribs:
            all_contribs[sensor] = {'vae': vae_errors.get(sensor, 0), 'shap': shap_vals.get(sensor, 0), 'kalman': innov, 'arima': arima_res.get(sensor, 0), 'total': innov}
    for sensor, res in sorted(arima_res.items(), key=lambda x: x[1], reverse=True)[:3]:
        if sensor not in all_contribs:
            all_contribs[sensor] = {'vae': vae_errors.get(sensor, 0), 'shap': shap_vals.get(sensor, 0), 'kalman': kalman_innov.get(sensor, 0), 'arima': res, 'total': res}
    combined_sorted = sorted(all_contribs.items(), key=lambda x: x[1]['total'], reverse=True)[:5]
    detailed_explanation = []
    sensor_names = state.vae_config['sensor_cols']
    for sensor_name, contribs in combined_sorted:
        idx = sensor_names.index(sensor_name) if sensor_name in sensor_names else 0
        actual_val = float(state.last_window_actual[idx]) if state.last_window_actual is not None else 0
        recon_val = float(state.last_window_recon[idx]) if state.last_window_recon is not None else 0
        deviation = abs(actual_val - recon_val)
        pct_deviation = (deviation / (abs(actual_val) + 0.001)) * 100
        reasoning = f"VAE err:{contribs['vae']:.3f} + SHAP:{abs(contribs['shap']):.3f} + Kal:{contribs['kalman']:.3f} + ARIMA:{contribs['arima']:.3f} → {sensor_name} anomaly."
        status = 'CRITICAL' if pct_deviation > 15 or contribs['total'] > 0.1 else 'WARNING' if pct_deviation > 5 else 'MINOR'
        detailed_explanation.append({
            'sensor': sensor_name,
            'vae_error': contribs['vae'],
            'shap_value': contribs['shap'],
            'kalman_innov': contribs['kalman'],
            'arima_res': contribs['arima'],
            'total_contrib': contribs['total'],
            'actual': actual_val,
            'reconstructed': recon_val,
            'status': status,
            'deviation_pct': pct_deviation,
            'reasoning': reasoning
        })
    return {
        'overall_score': current_anomaly['vae_score'],
        'threshold': state.vae_config.get('anomaly_threshold', 0.005),
        'vae_analysis': vae_analysis,
        'top_features': detailed_explanation,
        'all_sensor_names': list(vae_errors.keys()),
        'all_vae_errors': list(vae_errors.values()),
        'all_shap_values': list(shap_vals.values()) if shap_vals else [],
        'all_kalman_innov': list(kalman_innov.values()),
        'all_arima_res': list(arima_res.values())
    }

@app.route('/api/explainability')
def get_explainability():
    payload = build_explainability_payload()
    if payload is None:
        return jsonify({'error': 'No data available'}), 404
    return jsonify(payload)

@app.route('/api/stream')
def stream_updates():
    def event_stream():
        last_cycle = -1
        while True:
            try:
                if len(state.sensor_data_history) > 0:
                    current_cycle = state.current_cycle
                    if current_cycle != last_cycle:
                        sensors = list(state.sensor_data_history)[-1]
                        anomaly = list(state.anomaly_history)[-1] if len(state.anomaly_history) > 0 else None
                        # Provide explainability whenever buffer full (window completed)
                        explain = None
                        if len(state.sensor_buffer) == state.vae_config.get('window_size', 30):
                            explain = build_explainability_payload()
                        payload = {
                            'engine': state.current_engine,
                            'cycle': current_cycle,
                            'buffer_fill': len(state.sensor_buffer),
                            'sensors': sensors,
                            'anomaly': anomaly,
                            'explainability': explain
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        last_cycle = current_cycle
                time.sleep(0.5)
            except GeneratorExit:
                break
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
                time.sleep(1)
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)