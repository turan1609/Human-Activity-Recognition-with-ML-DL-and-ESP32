import serial
import time
import numpy as np
import joblib
import os
import sys
import warnings
import msvcrt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

COM_PORT = "COM3"
BAUD_RATE = 2000000
WINDOW_SIZE = 120
STEP = 60

LABEL_NAMES = ['Biceps_Curl', 'Dumbbell_Shoulder_Shrug', 'Front_Raise', 'Lateral_Raise', 'Sitting', 'Walking']

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def extract_features_classic(window_data):
    mean_vals = np.mean(window_data, axis=0)
    std_vals = np.std(window_data, axis=0)
    max_vals = np.max(window_data, axis=0)
    min_vals = np.min(window_data, axis=0)
    features = np.concatenate([mean_vals, std_vals, max_vals, min_vals])
    return features.reshape(1, -1)

def prepare_data_dl(window_data, scaler):
    scaled_data = scaler.transform(window_data)
    return scaled_data.reshape(1, WINDOW_SIZE, 6)

def main():
    clear_screen()
    print("="*60)
    print("   REAL-TIME HUMAN ACTIVITY RECOGNITION SYSTEM")
    print("="*60)
    print("\nSelect Model:\n")
    print(" [1] Random Forest")
    print(" [2] XGBoost")
    print(" [3] AdaBoost")
    print(" [4] CNN")
    print(" [5] LSTM")
    print("\n [Q] Quit")
    print("="*60)

    choice = input("\nEnter selection (1-5): ").strip().upper()

    if choice == 'Q':
        sys.exit()

    model_path = ""
    model_type = ""
    display_name = ""

    if choice == '1':
        model_path = "Random_Forest_model.pkl"
        model_type = "classic"
        display_name = "Random Forest"
    elif choice == '2':
        model_path = "XGBoost_model.pkl"
        model_type = "classic"
        display_name = "XGBoost"
    elif choice == '3':
        model_path = "AdaBoost_model.pkl"
        model_type = "classic"
        display_name = "AdaBoost"
    elif choice == '4':
        model_path = "cnn_model.h5"
        model_type = "dl"
        display_name = "CNN"
    elif choice == '5':
        model_path = "lstm_model.h5"
        model_type = "dl"
        display_name = "LSTM"
    else:
        print("Invalid selection.")
        sys.exit()

    print(f"\nLoading {display_name}...")

    model = None
    scaler = None

    try:
        if model_type == "classic":
            model = joblib.load(model_path)
        else:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load("scaler.pkl")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()

    print(f"Connecting to {COM_PORT}...")

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
    except Exception as e:
        print(f"Connection Error: {e}")
        sys.exit()

    buffer = []
    print("System Ready. Press 'Q' to exit.")

    try:
        while True:
            if msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    break

            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue
                    
                    parts = line.split(',')
                    if len(parts) != 6: continue
                    
                    vals = [float(x) for x in parts]
                    buffer.append(vals)

                    if len(buffer) == WINDOW_SIZE:
                        data_np = np.array(buffer)
                        prediction_label = ""
                        confidence_display = ""

                        if model_type == "classic":
                            features = extract_features_classic(data_np)
                            pred_id = model.predict(features)[0]
                            prediction_label = LABEL_NAMES[pred_id]
                        else:
                            input_data = prepare_data_dl(data_np, scaler)
                            pred_probs = model.predict(input_data, verbose=0)
                            pred_id = np.argmax(pred_probs, axis=1)[0]
                            confidence = np.max(pred_probs) * 100
                            prediction_label = LABEL_NAMES[pred_id]
                            confidence_display = f"({confidence:.1f}%)"

                        clear_screen()
                        print("\n" * 2)
                        print("=" * 50)
                        print(f"   ACTIVE MODEL: {display_name}")
                        print("=" * 50)
                        print("\n\n")
                        print(f"   DETECTED ACTIVITY:")
                        print("\n")
                        print(f"   >>> {prediction_label.upper()} <<<")
                        print("\n")
                        if confidence_display:
                            print(f"       {confidence_display}")
                        print("\n\n")
                        print("=" * 50)
                        print("Press 'Q' to exit")

                        buffer = buffer[STEP:]

                except ValueError:
                    continue

    except KeyboardInterrupt:
        pass
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()