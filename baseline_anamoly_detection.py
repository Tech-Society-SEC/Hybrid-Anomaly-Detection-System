import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore', 'statsmodels.tsa.arima.model.ARIMA', UserWarning)

FILENAME = 'synthetic_sensor_data.csv'

def calculate_anomaly_score(original, filtered):
    """Calculates anomaly score as the absolute difference."""
    return np.abs(original - filtered)

def run_kalman_filter(series):
    """
    Applies a Kalman Filter to a time series to smooth it and detect anomalies.
    """
    print("Running Kalman Filter...")
    kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(series.values)
    return pd.Series(filtered_state_means.flatten(), index=series.index)

def run_arima_forecast(series, order=(5, 1, 0)):
    """
    Uses a rolling ARIMA forecast to predict the next step and find anomalies.
    This is computationally intensive.
    """
    print("Running rolling ARIMA forecast (this may take a moment)...")
    history = [x for x in series]
    predictions = []
    
    # Use a rolling window for prediction
    window = 200
    for t in range(len(series)):
        if t < window:
            predictions.append(series.iloc[t]) # Can't predict for the initial period
            continue
        
        model = ARIMA(history[-window:], order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(series.iloc[t])
        
    print("ARIMA forecast complete.")
    return pd.Series(predictions, index=series.index)


def main():
    """
    Main function to load data, run models, and plot results.
    """
    try:
        df = pd.read_csv(FILENAME, index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {FILENAME} not found. Please run eda.py first to generate the data.")
        return

    # We will analyze Sensor_4 for this example, as it has a clear anomaly
    sensor_series = df['Sensor_4']

    # --- Run Models ---
    kf_smoothed = run_kalman_filter(sensor_series)
    arima_predictions = run_arima_forecast(sensor_series)
    
    # --- Calculate Anomaly Scores ---
    kf_anomaly_scores = calculate_anomaly_score(sensor_series, kf_smoothed)
    arima_anomaly_scores = calculate_anomaly_score(sensor_series, arima_predictions)
    
    # --- Plotting ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
    
    # Plot 1: Original Data vs. Model Fits
    axs[0].plot(sensor_series, label='Original Sensor Data', color='blue', alpha=0.7)
    axs[0].plot(kf_smoothed, label='Kalman Filter Smoothed', color='orange', linestyle='--')
    axs[0].plot(arima_predictions, label='ARIMA Rolling Forecast', color='green', linestyle=':')
    axs[0].set_title('Sensor 4 Data vs. Baseline Model Fits')
    axs[0].legend()
    axs[0].set_ylabel('Sensor Value')

    # Plot 2: Kalman Filter Anomaly Score
    axs[1].plot(kf_anomaly_scores, label='Kalman Filter Anomaly Score', color='orange')
    axs[1].set_title('Kalman Filter Anomaly Score (Absolute Error)')
    axs[1].legend()
    axs[1].set_ylabel('Score')

    # Plot 3: ARIMA Anomaly Score
    axs[2].plot(arima_anomaly_scores, label='ARIMA Anomaly Score', color='green')
    axs[2].set_title('ARIMA Anomaly Score (Absolute Error)')
    axs[2].legend()
    axs[2].set_ylabel('Score')
    axs[2].set_xlabel('Timestamp')

    plt.tight_layout()
    plt.savefig('baseline_model_results.png')
    plt.close()
    
    print("Plot saved to baseline_model_results.png")

if __name__ == '__main__':
    main()

