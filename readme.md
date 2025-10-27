# Hybrid Time Series Anomaly Detection for Predictive Maintenance

An interactive dashboard for real-time anomaly detection in industrial sensor data using a hybrid of classical statistical models and modern deep learning techniques, with a focus on Explainable AI (XAI).

---


## Note : As we have directly commited to the main branch rather a feature branch, we are updating the changes in the readme file without a PR.

```
PR Summary:

This is a major feature update that completes the core objective of the project, moving it into Phase 3. The dashboard is no longer a single-model prototype; it is now a fully functional, hybrid system that runs the VAE, ARIMA, and Kalman Filter models in parallel.

This PR refactors the baseline model script to save trained models and upgrades the Flask backend to load and serve live anomaly scores from all three models simultaneously. The frontend now provides a real-time comparative view of how both classical and deep learning models interpret the sensor data, fulfilling the project's primary goal.

Team Roles & Contributions:

Member

Role

Work Contribution

Karnala Santhan Kumar

Author & System Integrator

Led the development by refactoring the baseline_models.py script to train and save the classical models. He upgraded the core flask.py backend to handle the loading and parallel processing of all three models.

Hanshika Varthini R

Reviewer & Validation Engineer

Reviewed the new architecture to ensure all models were integrated correctly without performance issues. She was responsible for validating the end-to-end workflow and confirming that all three gauges on the dashboard displayed accurate, live data.

Key Changes:

1. baseline_models.py (Refactored from baseline_anamoly_detection.py)

Change of Purpose: The script no longer just performs analysis. Its new role is to train and serialize the baseline models so they can be loaded instantly by the application.

Model Training: The script now trains the Kalman Filter and fits the ARIMA model on the "normal" training data (the first 80% of the dataset).

Model Saving:

The trained Kalman Filter object is saved to kf_model.pkl.

The ARIMA configuration (its order and a calculated anomaly threshold) is saved to arima_config.pkl.

2. flask.py (Major Upgrade)

Loads All Models: On startup, the load_dependencies function now loads vae_model.h5, kf_model.pkl, and arima_config.pkl.

Parallel Inference in API (/api/anomaly-scores): This endpoint has been completely rebuilt to:

Run VAE: Performs multivariate anomaly detection on a window of all sensors.

Run Kalman Filter: Applies the loaded filter to the target sensor's history and calculates an anomaly score based on the error.

Run ARIMA: Performs a live, rolling forecast on the target sensor and calculates an anomaly score based on the prediction error.

Comparative Scores: The API response now includes three distinct keys: vae, kalman, and arima, each with a normalized score. This directly feeds the three gauges on the dashboard.

How to Test and Verify:

IMPORTANT: First, rename the old file baseline_anamoly_detection.py to baseline_models.py.

Run the model preparation scripts in order to ensure all model files are created:

# 1. Generate the data if you haven't already
python eda.py

# 2. Train and save the VAE model
python vae_model.py

# 3. Train and save the NEW baseline models
python baseline_models.py


After this step, you should have vae_model.h5, kf_model.pkl, and arima_config.pkl in your directory.

Launch the fully integrated dashboard:

python flask.py
```

## The model files are yet to be pushed! but the code changes are reflected with proper changes!

## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Project Timeline](#project-timeline)
- [License](#license)

---

## About The Project

Industrial equipment failures lead to costly downtime and significant safety risks. Traditional maintenance schedules are often inefficient, reacting to failures rather than preventing them. While modern predictive maintenance methods exist, they face their own challenges:
- **Statistical models** (like ARIMA) struggle with the complex, non-linear relationships in multivariate sensor data.
- **Deep learning models** (like VAEs) can be powerful but often operate as "black boxes," making it difficult for engineers to trust their outputs and take action.

This project addresses these gaps by creating a robust, hybrid system that not only proactively detects anomalies in sensor data but also provides clear, actionable explanations for its decisions. The final deliverable is a functional software prototype featuring a real-time, interactive dashboard that compares multiple models and pinpoints the root cause of failures.

---

## Key Features

- **Hybrid Modeling Approach:** Integrates and compares three distinct models in real-time:
  - **ARIMA:** A foundational statistical model for baseline comparison.
  - **Kalman Filter:** A state-space model to smooth data and identify deviations.
  - **Variational Autoencoder (VAE):** A deep learning model for capturing complex patterns in normal operational data.
- **Explainable AI (XAI) Module:** The core innovation of this project. When the VAE detects an anomaly, the system analyzes the **per-sensor reconstruction error** to identify which specific sensor(s) are behaving abnormally, providing a root-cause analysis.
- **Interactive Real-Time Dashboard:** A user-friendly web interface built with Flask and Chart.js to visualize:
  - Live sensor data streams.
  - Comparative anomaly scores from all three models.
  - Actionable alerts and explanations from the XAI module.
- **End-to-End Pipeline:** Demonstrates a full pipeline from data analysis and model training to deployment in a web-based application.

---

## System Architecture

The system is designed with a simple, scalable architecture that separates data processing, modeling, and presentation.

1.  **Data Source:** The system uses the NASA C-MAPSS dataset, with a synthetic data generator included for rapid prototyping and testing.
2.  **Backend (Flask):** A Python-based server that manages model inference and provides a REST API for the frontend.
    - `/api/sensor-data`: Streams the latest sensor readings.
    - `/api/anomaly-scores`: Provides real-time anomaly scores from the VAE, ARIMA, and Kalman Filter models.
    - `/api/xai-analysis`: Delivers the root-cause analysis when an anomaly is detected.
3.  **Modeling Engine:** A collection of Python scripts containing the trained models. The backend calls these models to process new data points.
4.  **Frontend (HTML/CSS/JS):** A single-page dashboard that polls the backend APIs and dynamically updates charts and status indicators to provide a live view of the system's health.

---

## Technologies Used

| Category          | Technology                                                                                                                                                                                          |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) |
| **ML/Statistics** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) `statsmodels`, `pykalman` |
| **Data Science** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white) |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) ![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black) ![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white) |

---

## Getting Started

Following these instructions will get a local copy of the project up for running.

### Prerequisites

- Python 3.8+
- pip (Python package installer)
- Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Tech-Society-SEC/Hybrid-Anomaly-Detection-System.git
    cd Hybrid-Anomaly-Detection-System
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is included for easy installation.
    ```sh
    pip install -r requirements.txt
    ```

---

## Usage

1.  **Generate Data & Perform Initial Analysis:**
    Run the exploratory data analysis script. This will generate the `synthetic_sensor_data.csv` file used by the application and save several analytical plots.
    ```sh
    python eda.py
    ```

2.  **Run the Baseline Model Analysis (Optional):**
    To see how the traditional models perform on the dataset, run this script. It will generate a plot comparing the ARIMA and Kalman Filter results.
    ```sh
    python baseline_models.py
    ```

3.  **Launch the Dashboard:**
    Start the Flask web server.
    ```sh
    python app.py
    ```

4.  **View the Dashboard:**
    Open your web browser and navigate to `http://127.0.0.1:5001`. You should see the live dashboard interface.

---


