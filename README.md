# Real-Time Human Activity Recognition (HAR) with ESP32 & Deep Learning

This project implements a real-time **Human Activity Recognition** system using IMU sensor data collected via an **ESP32/Bluetooth** module. The system systematically evaluates and compares 5 different Machine Learning and Deep Learning models to classify movements instantly.

## 🎥 Project Demo

<!-- AŞAĞIDAKİ ALANA VİDEONU SÜRÜKLEYİP BIRAK, GITHUB OTOMATİK LİNK OLUŞTURACAK -->
Here is the system in action, classifying movements in real-time:


https://github.com/user-attachments/assets/c1853d3e-992e-46ce-8476-28d86ffa8dde


## 🛠️ Hardware Setup

The data is collected using a wireless IMU sensor setup (Accelerometer & Gyroscope).

![WhatsApp Image 2026-01-11 at 21 40 27](https://github.com/user-attachments/assets/f2692f05-67c2-4fd8-9f91-a8a8b33f6298)

- **Device:** ESP32 / Custom IMU Module
- **Communication:** Serial / Bluetooth (2,000,000 Baud Rate)
- **Sensor Data:** 6-axis (Acc X, Y, Z + Gyro X, Y, Z)

## 🚀 Features

- **Real-Time Inference:** Classifies activities instantly using a sliding window approach (Window Size: 120).
- **Multi-Model Support:** Capable of running 5 different models switchable via a menu.
- **Data Preprocessing:** Automatic noise filtering and normalization.
- **GUI-Compatible Code:** Designed to work with shared graphical interfaces.

## 📊 Model Performance Comparison

We trained and evaluated the following models on the collected dataset.

| Model | Type | Accuracy | Latency (s) | Best Use Case |
|-------|------|----------|-------------|---------------|
| **XGBoost** | Classic ML | **~94%** | Very Low | **Recommended (Balanced)** |
| **CNN** | Deep Learning | ~93% | Low | High Accuracy Feature Extraction |
| **Random Forest** | Classic ML | ~91% | Low | Robust Baseline |
| **LSTM** | Deep Learning | ~85% | Medium | Temporal Dependency |
| **AdaBoost** | Classic ML | ~73% | Very Low | Fast Inference |

## 📂 Project Structure

- `Human_Activity_Recognition.ipynb`: Complete training pipeline (Preprocessing -> Training -> Evaluation).
- `Real_Time_Test.py`: The main script for connecting to the device and testing models in real-time.
- `models/`: Contains trained `.pkl` and `.h5` files.
- `scaler.pkl`: StandardScaler object required for Deep Learning models.
- `human_activity/`: Dataset folder containing CSV files.

## 💻 Installation & Usage

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/turan1609/Human-Activity-Recognition-with-ML-DL-and-ESP32.git
    cd Human-Activity-Recognition-with-ML-DL-and-ESP32
    ```

2.  **Install Requirements:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow joblib pyserial
    ```

3.  **Run Real-Time Test:**
    Connect your device to USB/Bluetooth and check the COM port (Update `COM_PORT` in the script if needed).
    ```bash
    python Real_Time_Test.py
    ```
    *Select a model from the menu (1-5) and start moving!*

---
