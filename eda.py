import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft

# --- Configuration ---
NUM_SAMPLES = 2000
NUM_SENSORS = 5
FILENAME = 'synthetic_sensor_data.csv'

def generate_synthetic_data():
    """
    Generates a synthetic multivariate time series dataset for predictive maintenance.

    The data includes:
    - A stable base signal for each sensor.
    - Cyclical patterns (seasonality) to simulate machine operations.
    - A gradual trend to simulate wear and tear.
    - Random noise to mimic real-world sensor fluctuations.
    - A sudden, sharp anomaly injected towards the end of the series.
    """
    print("Generating synthetic sensor data...")
    time = np.arange(NUM_SAMPLES)
    
    # Base signals for each sensor
    sensor_data = np.zeros((NUM_SAMPLES, NUM_SENSORS))
    base_levels = [100, 50, 200, 150, 75]
    
    for i in range(NUM_SENSORS):
        # 1. Seasonality (cyclical patterns)
        seasonality = 10 * np.sin(2 * np.pi * time / (100 + i*20)) + \
                      5 * np.cos(2 * np.pi * time / (50 - i*5))

        # 2. Trend (gradual wear)
        trend = base_levels[i] + 0.01 * time * (i + 1)

        # 3. Noise
        noise = np.random.normal(0, 2, NUM_SAMPLES)
        
        sensor_data[:, i] = trend + seasonality + noise

    # 4. Inject Anomaly
    anomaly_start = int(NUM_SAMPLES * 0.85)
    anomaly_length = int(NUM_SAMPLES * 0.05)
    
    # The anomaly affects sensors 1 and 3 more significantly
    sensor_data[anomaly_start : anomaly_start + anomaly_length, 1] += 30 * np.sin(np.linspace(0, np.pi, anomaly_length))
    sensor_data[anomaly_start : anomaly_start + anomaly_length, 3] += 25 * (1 - np.cos(np.linspace(0, 2*np.pi, anomaly_length)))
    sensor_data[anomaly_start : anomaly_start + anomaly_length, :] += np.random.normal(0, 5, (anomaly_length, NUM_SENSORS))
    print(f"Anomaly injected from index {anomaly_start} to {anomaly_start + anomaly_length}.")

    # Create DataFrame
    columns = [f'Sensor_{i+1}' for i in range(NUM_SENSORS)]
    df = pd.DataFrame(sensor_data, columns=columns)
    df['Timestamp'] = pd.to_datetime(pd.date_range(start='2025-01-01', periods=NUM_SAMPLES, freq='min'))
    df = df.set_index('Timestamp')
    
    df.to_csv(FILENAME)
    print(f"Data saved to {FILENAME}")
    return df

def perform_eda(df):
    """
    Performs and saves key exploratory data analysis plots.
    """
    print("Performing Exploratory Data Analysis...")
    
    # 1. Plot the raw sensor data
    plt.style.use('seaborn-v0_8-whitegrid')
    df.plot(subplots=True, figsize=(15, 12), title='Synthetic Sensor Data Over Time')
    plt.xlabel('Timestamp')
    plt.tight_layout()
    plt.savefig('sensor_data_plot.png')
    plt.close()
    print("Saved sensor_data_plot.png")

    # 2. Time Series Decomposition (on a single sensor)
    # Decompose Sensor 1 to show trend, seasonality, and residuals
    decomposition = seasonal_decompose(df['Sensor_1'], model='additive', period=100)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle('Time Series Decomposition of Sensor 1', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('decomposition_plot.png')
    plt.close()
    print("Saved decomposition_plot.png")

    # 3. Spectral Analysis (on a single sensor)
    # Use Fast Fourier Transform (FFT) to find dominant frequencies
    sensor_1_data = df['Sensor_1'].values
    N = len(sensor_1_data)
    T = 1.0 # 1 minute sampling
    yf = fft(sensor_1_data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    plt.figure(figsize=(14, 7))
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title('Spectral Analysis of Sensor 1')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('spectral_analysis_plot.png')
    plt.close()
    print("Saved spectral_analysis_plot.png")

if __name__ == '__main__':
    data = generate_synthetic_data()
    perform_eda(data)
    print("\nEDA complete. Check the generated CSV and plot images.")

