import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error
from veri_on_isleme import veri_on_islem


def lstm_model_egit(data, window_size=30, epochs=50, test_ratio=0.10, output_dir="outputs", plot_dir="static/plots"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, df_clean = veri_on_islem(data, window_size=window_size, test_ratio=test_ratio)
    split_val = int(len(X_test) * 0.5)
    if split_val == 0 or len(X_test) - split_val == 0:
        raise ValueError("Validation/Test ayırımı için yeterli test verisi yok.")

    X_val = X_test[:split_val]
    y_val = y_test[:split_val]

    X_test = X_test[split_val:]
    y_test = y_test[split_val:]

    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(128, return_sequences=True), 
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"), 
        Dense(1)])

    model.compile(optimizer="adam", loss="huber")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
                            X_train,
                            y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=64,
                            shuffle=False,
                            callbacks=[early_stop],
                            verbose=1
                            )

    y_pred = model.predict(X_test, verbose=0)

    
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred)

    # Güvenlik: NaN, inf ve negatif değerleri temizle
    y_test_real = np.nan_to_num(y_test_real, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_real = np.nan_to_num(y_pred_real, nan=0.0, posinf=0.0, neginf=0.0)

    y_test_real = np.maximum(0, y_test_real)
    y_pred_real = np.maximum(0, y_pred_real)

    # 1D hale getir
    y_test_real = y_test_real.flatten()
    y_pred_real = y_pred_real.flatten()

    # Metrikler
    mae = mean_absolute_error(y_test_real, y_pred_real)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    # MAPE: gerçek değer 0 olanlarda bölme hatasını önlemek için max(1, y_true)
    mask = y_test_real != 0
    mape = np.mean(np.abs((y_test_real[mask] - y_pred_real[mask])/ y_test_real[mask])) * 100
    
    model_path = os.path.join(output_dir, "lstm_talep_model.h5")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Loss grafiği
    loss_plot_path = os.path.join(plot_dir, "lstm_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Eğitim Loss")
    plt.plot(history.history["val_loss"], label="Doğrulama Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("LSTM Eğitim ve Doğrulama Loss Grafiği")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    # Tahmin grafiği
    prediction_plot_path = os.path.join(plot_dir, "lstm_prediction.png")

    n_plot = min(200, len(y_test_real))

    plt.figure(figsize=(8, 5))
    plt.plot(y_test_real[:n_plot], label="Gerçek Talep")
    plt.plot(y_pred_real[:n_plot], label="Tahmin Edilen Talep")
    plt.xlabel("Gözlem")
    plt.ylabel("Talep")
    plt.title("Gerçek ve Tahmin Edilen Talep Karşılaştırması")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(prediction_plot_path)
    plt.close()

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "epoch_count": int(len(history.history["loss"])),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "loss_plot": loss_plot_path,
        "prediction_plot": prediction_plot_path
    }


