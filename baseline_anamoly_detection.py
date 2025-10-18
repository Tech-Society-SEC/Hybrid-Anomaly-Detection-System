import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima.model.ARIMA', UserWarning)

# --- Configuration ---
FILENAME = 'synthetic_sensor_data.csv'
TRAIN_SPLIT_PERCENT = 0.8 # Use the same split as the VAE for consistency
ARIMA_ORDER = (5, 1, 0)
TARGET_SENSOR = 'Sensor_4' # The sensor we will model with classical methods

def train_and_save_models():
    """
    Trains the ARIMA and Kalman Filter models on the 'normal' training data
    and saves the trained models to disk using pickle.
    """
    print("--- Training and Saving Baseline Models ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(FILENAME, index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {FILENAME} not found. Please run eda.py first.")
        return

    # 2. Split into training data (normal operation)
    train_size = int(len(df) * TRAIN_SPLIT_PERCENT)
    train_series = df[TARGET_SENSOR].iloc[:train_size]
    
    # --- 3. Train and Save Kalman Filter ---
    print(f"Training Kalman Filter on '{TARGET_SENSOR}'...")
    # Use Expectation-Maximization (EM) to 'train' the filter on the normal data
    kf = KalmanFilter(initial_state_mean=train_series.iloc[0], n_dim_obs=1)
    kf = kf.em(train_series.values, n_iter=10) 
    
    with open('kf_model.pkl', 'wb') as f:
        pickle.dump(kf, f)
    print("Kalman Filter model saved to kf_model.pkl")

    # --- 4. Fit ARIMA and Save Config ---
    # We save the model's configuration and a calculated error threshold.
    # The model itself will be refit on-the-fly in the Flask app.
    print(f"Fitting ARIMA model on '{TARGET_SENSOR}' to determine error threshold...")
    model = ARIMA(train_series, order=ARIMA_ORDER)
    model_fit = model.fit()
    
    # Calculate residuals (errors) on the training data
    train_residuals = model_fit.resid
    
    # An anomaly threshold can be a multiple of the standard deviation of these errors
    # Using a 3-sigma rule is a common statistical baseline
    arima_threshold = np.std(train_residuals) * 3 
    
    arima_config = {'order': ARIMA_ORDER, 'threshold': arima_threshold}
    
    with open('arima_config.pkl', 'wb') as f:
        pickle.dump(arima_config, f)
    print(f"ARIMA config saved to arima_config.pkl (Threshold: {arima_threshold:.4f})")
    print("--- Baseline models prepared successfully! ---")


if __name__ == '__main__':
    train_and_save_models()
