import pandas as pd
import numpy as np

# --- Configuration ---
RAW_DATA_FILE = 'train_FD001.txt' # The file from the NASA zip
NORMAL_CYCLES_THRESHOLD = 50      # Use the first 50 cycles of each engine as "normal"
STREAM_ENGINE_ID = 3              # We'll pick Engine #3 to be our live stream test case

# Column names from the C-MAPSS readme.txt
COLUMN_NAMES = [
    'engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

def process_cmaps_data():
    """
    Loads the raw C-MAPSS data, processes it, and saves two CSV files:
    1. train_data_normal.csv: Data from the first 50 cycles of all engines.
    2. test_engine_stream.csv: Full data from a single engine (Engine #3).
    """
    print(f"Loading raw data from {RAW_DATA_FILE}...")
    try:
        raw_df = pd.read_csv(RAW_DATA_FILE, sep=' ', header=None)
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"File not found: {RAW_DATA_FILE}")
        print("Please download the C-MAPSS dataset (FD001) and place the file in this directory.")
        print("Download link: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository")
        print("---------------\n")
        return

    # Drop the last two empty columns that pandas reads from the space delimiter
    raw_df.drop(columns=[26, 27], inplace=True)
    raw_df.columns = COLUMN_NAMES
    
    print("Raw data loaded. Identifying constant-value columns to drop...")

    # Find columns that have constant values (uninformative)
    constant_cols = [col for col in raw_df.columns if raw_df[col].std() == 0]
    print(f"Dropping constant columns: {constant_cols}")
    
    # Also drop the settings columns, we only want sensor data
    cols_to_drop = constant_cols + ['setting_1', 'setting_2', 'setting_3']
    processed_df = raw_df.drop(columns=cols_to_drop)

    # --- 1. Create the "Normal" Training Set ---
    # Group by engine and take the first NORMAL_CYCLES_THRESHOLD cycles
    print(f"Extracting first {NORMAL_CYCLES_THRESHOLD} cycles from each engine for 'normal' training set...")
    normal_data_frames = []
    for engine_id in processed_df['engine_id'].unique():
        engine_data = processed_df[processed_df['engine_id'] == engine_id]
        early_cycles = engine_data.head(NORMAL_CYCLES_THRESHOLD)
        normal_data_frames.append(early_cycles)
        
    normal_df = pd.concat(normal_data_frames)
    
    # For training, we don't need engine_id or cycle
    normal_df_sensors_only = normal_df.drop(columns=['engine_id', 'cycle'])
    
    # Save the normal training data
    normal_df_sensors_only.to_csv('train_data_normal.csv', index=False)
    print(f"Successfully saved train_data_normal.csv (Shape: {normal_df_sensors_only.shape})")

    # --- 2. Create the "Streaming" Test Set ---
    print(f"Extracting full data for Engine ID {STREAM_ENGINE_ID} for 'streaming' test set...")
    stream_df = processed_df[processed_df['engine_id'] == STREAM_ENGINE_ID].copy()
    
    # For the stream, we want to keep the cycle and engine_id, but drop them for the models
    # We will save the sensor data only, just like the training set
    stream_df_sensors_only = stream_df.drop(columns=['engine_id', 'cycle'])
    
    # Save the streaming test data
    stream_df_sensors_only.to_csv('test_engine_stream.csv', index=False)
    print(f"Successfully saved test_engine_stream.csv (Shape: {stream_df_sensors_only.shape})")
    print("\nC-MAPSS data processing complete.")

if __name__ == '__main__':
    process_cmaps_data()
