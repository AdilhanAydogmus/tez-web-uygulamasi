import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from veri_on_isleme import veri_on_islem


def lstm_model_egit(
    data,
    window_size=30,
    epochs=20,
    output_dir="outputs",
    plot_dir="static/plots"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, df_clean = veri_on_islem(
        data,
        window_size=window_size
    )

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(window_size, 1)
        )
    )
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.10,
        epochs=epochs,
        batch_size=128,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test)

    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred)

    y_test_real = np.expm1(y_test_real)
    y_pred_real = np.expm1(y_pred_real)

    y_test_real = np.nan_to_num(y_test_real, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_real = np.nan_to_num(y_pred_real, nan=0.0, posinf=0.0, neginf=0.0)

    mae = mean_absolute_error(y_test_real, y_pred_real)

    rmse = np.sqrt(
        mean_squared_error(y_test_real, y_pred_real)
    )

    mape = np.mean(
        np.abs((y_test_real - y_pred_real) / np.maximum(1, y_test_real))
    ) * 100

    wape = (
        np.sum(np.abs(y_test_real - y_pred_real))
        / np.sum(np.maximum(1, y_test_real))
    ) * 100

    model_path = os.path.join(output_dir, "lstm_talep_model.h5")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    loss_plot_path = os.path.join(plot_dir, "lstm_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Eğitim Loss")
    plt.plot(history.history["val_loss"], label="Doğrulama Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Eğitim ve Doğrulama Loss Grafiği")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    prediction_plot_path = os.path.join(plot_dir, "lstm_prediction.png")

    plt.figure(figsize=(8, 5))
    plt.plot(y_test_real[:200], label="Gerçek Talep")
    plt.plot(y_pred_real[:200], label="Tahmin Edilen Talep")
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
        "wape": float(wape),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "epoch_count": int(len(history.history["loss"])),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "loss_plot": loss_plot_path,
        "prediction_plot": prediction_plot_path
    }