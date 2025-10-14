```python 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# --- Configuration ---
FILENAME = 'synthetic_sensor_data.csv'
TIME_STEPS = 10  # How many previous time steps to use for predicting the next one
TRAIN_SPLIT_PERCENT = 0.8  # Use the first 80% of data for training (assumed normal)
LATENT_DIM = 2 # Number of latent variables in the VAE

# --- 1. Data Preparation ---

def load_and_preprocess_data():
    """Loads, scales, and creates sequences from the sensor data."""
    try:
        df = pd.read_csv(FILENAME, index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {FILENAME} not found. Please run eda.py first.")
        return None, None

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Create sequences of data
    X = []
    for i in range(len(df_scaled) - TIME_STEPS):
        X.append(df_scaled[i:(i + TIME_STEPS)])
    
    return np.array(X), scaler

# --- 2. VAE Model Architecture ---

def build_vae(n_features):
    """Builds the Variational Autoencoder model."""
    # Encoder
    inputs = Input(shape=(TIME_STEPS, n_features))
    # Flatten the input for Dense layers
    flatten = tf.keras.layers.Flatten()(inputs)
    h = Dense(64, activation='relu')(flatten)
    h = Dense(32, activation='relu')(h)
    
    # Latent space
    z_mean = Dense(LATENT_DIM)(h)
    z_log_sigma = Dense(LATENT_DIM)(h)

    # Sampling function (reparameterization trick)
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling)([z_mean, z_log_sigma])
    
    # Decoder
    decoder_h = Dense(32, activation='relu')
    decoder_mean = Dense(TIME_STEPS * n_features, activation='sigmoid') # Sigmoid for scaled data
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    # Reshape the output to match the input shape
    reshaped_output = tf.keras.layers.Reshape((TIME_STEPS, n_features))(x_decoded_mean)

    # VAE model
    vae = Model(inputs, reshaped_output)
    
    # VAE loss
    reconstruction_loss = tf.keras.losses.mse(flatten, x_decoded_mean)
    reconstruction_loss *= TIME_STEPS * n_features
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    vae.compile(optimizer='adam')
    return vae

# --- 3. Training and Evaluation ---

def main():
    """Main function to run the VAE training pipeline."""
    X, scaler = load_and_preprocess_data()
    if X is None:
        return

    # Split data into training (normal) and testing (contains anomaly)
    train_size = int(len(X) * TRAIN_SPLIT_PERCENT)
    X_train = X[:train_size]
    X_test = X[train_size:]
    
    n_features = X.shape[2]
    
    print("Building and compiling VAE model...")
    vae = build_vae(n_features)
    vae.summary()
    
    print("\nTraining VAE on normal data...")
    history = vae.fit(X_train, X_train,
                      epochs=50,
                      batch_size=32,
                      validation_split=0.1,
                      verbose=1)
                      
    # Save the trained model for use in the Flask app
    vae.save('vae_model.h5')
    print("\nModel trained and saved as vae_model.h5")
    
    # --- Visualize Reconstruction Error ---
    X_train_pred = vae.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=(1, 2))

    plt.figure(figsize=(12, 6))
    plt.hist(train_mae_loss, bins=50)
    plt.xlabel('Train MAE loss')
    plt.ylabel('Number of samples')
    plt.title('Distribution of Reconstruction Error on Normal Data')
    plt.savefig('vae_reconstruction_error_distribution.png')
    plt.show()
    print("Saved reconstruction error plot to vae_reconstruction_error_distribution.png")
    
    # Set anomaly threshold based on training data
    threshold = np.max(train_mae_loss) * 1.1 # A bit higher than the max error on normal data
    print(f"\nAnomaly threshold set to: {threshold:.4f}")

if __name__ == '__main__':
    main()
```
