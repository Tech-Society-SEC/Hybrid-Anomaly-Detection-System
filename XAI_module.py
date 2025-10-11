import numpy as np

def analyze_reconstruction_error(original_data, reconstructed_data, sensor_names):
    """
    Calculates the per-sensor contribution to the total reconstruction error.
    This is the core of the XAI module.

    Args:
        original_data (np.ndarray): A 1D array of the true sensor readings at a point in time.
        reconstructed_data (np.ndarray): A 1D array of the VAE's reconstructed sensor readings.
        sensor_names (list): A list of strings with the names of the sensors.

    Returns:
        dict: A dictionary with sensor names as keys and their percentage contribution
              to the total squared error as values. Sorted by contribution.
    """
    if original_data.shape != reconstructed_data.shape:
        raise ValueError("Input and reconstructed data must have the same shape.")

    # Calculate squared error for each sensor
    squared_errors = (original_data - reconstructed_data) ** 2
    
    # Calculate the total error
    total_error = np.sum(squared_errors)
    
    # If there is no error, all contributions are zero
    if total_error == 0:
        return {name: 0.0 for name in sensor_names}

    # Calculate the percentage contribution of each sensor
    error_contributions = (squared_errors / total_error) * 100
    
    # Create a dictionary for easy interpretation
    contributions_dict = dict(zip(sensor_names, error_contributions))
    
    # Sort the dictionary by contribution percentage (descending)
    sorted_contributions = {k: v for k, v in sorted(contributions_dict.items(), key=lambda item: item[1], reverse=True)}

    return sorted_contributions

# --- Example Usage (to be replaced with live data in app.py) ---
if __name__ == '__main__':
    # This is a simulation of an anomaly event.
    print("--- XAI Module Example ---")
    
    sensor_names = ['Temperature', 'Pressure', 'Vibration', 'Flow_Rate', 'Acoustic']

    # Example 1: A normal data point with very low reconstruction error
    normal_input = np.array([80.1, 150.5, 0.2, 5.5, 0.1])
    normal_recon = np.array([80.3, 150.2, 0.21, 5.45, 0.12])
    
    print("\nAnalyzing a NORMAL data point:")
    normal_analysis = analyze_reconstruction_error(normal_input, normal_recon, sensor_names)
    for sensor, contribution in normal_analysis.items():
        print(f"- {sensor}: {contribution:.2f}% contribution")

    # Example 2: An anomalous data point where Pressure and Flow_Rate are off
    anomalous_input = np.array([85.0, 250.0, 0.3, 2.1, 0.15]) # Pressure is very high, Flow Rate very low
    anomalous_recon = np.array([85.2, 155.0, 0.28, 5.2, 0.16]) # VAE tries to "correct" them to normal values
    
    print("\nAnalyzing an ANOMALOUS data point:")
    anomaly_analysis = analyze_reconstruction_error(anomalous_input, anomalous_recon, sensor_names)
    for sensor, contribution in anomaly_analysis.items():
        print(f"- {sensor}: {contribution:.2f}% contribution")
    
    print(f"\nConclusion: The XAI module correctly identifies 'Pressure' ({anomaly_analysis['Pressure']:.2f}%) and 'Flow_Rate' ({anomaly_analysis['Flow_Rate']:.2f}%) as the primary drivers of the anomaly.")

