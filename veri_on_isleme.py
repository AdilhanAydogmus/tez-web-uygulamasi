import pandas as pd
import numpy as np
import joblib

from veri_yukleme import veri_yukle
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view


def veri_on_islem(data, window_size: int = 30, test_ratio: float = 0.10):

    df = veri_yukle(data)

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["store", "date"])

    # Negatif güvenliği
    df["sales"] = np.maximum(df["sales"], 0)


    train_parts = []
    test_parts = []

    for store_id, temp in df.groupby("store"):

        temp = temp.sort_values("date").copy()

        if len(temp) <= window_size:
            continue

        split_index = int(len(temp) * (1 - test_ratio))

        if split_index <= window_size:
            continue

        train_part = temp.iloc[:split_index].copy()

        # Window continuity için overlap bırakıyoruz
        test_part = temp.iloc[split_index - window_size:].copy()

        train_parts.append(train_part)

        test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True)

    test_df = pd.concat(test_parts, ignore_index=True)

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler.fit(train_df[["sales"]])

    train_df["sales_scaled"] = scaler.transform(train_df[["sales"]])

    test_df["sales_scaled"] = scaler.transform(test_df[["sales"]])

    joblib.dump(scaler, "scaler.pkl")

    def create_windows(dataframe):
        X_list = []
        y_list = []
        for _, temp in dataframe.groupby("store"):

            values = temp["sales_scaled"].values

            if len(values) <= window_size:
                continue

            # Sliding Window
            windows = sliding_window_view(values, window_shape=window_size + 1)

            X_list.append(windows[:, :-1])
            y_list.append(windows[:, -1])

        X = np.vstack(X_list)

        y = np.concatenate(y_list)

        return X, y

    X_train, y_train = create_windows(train_df)

    X_test, y_test = create_windows(test_df)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    df_clean = pd.concat([train_df, test_df.groupby("store").apply(lambda x: x.iloc[window_size:])
            .reset_index(drop=True)], ignore_index=True ).sort_values(["store", "date"])

    print("--- Ön İşleme Tamamlandı ---")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        df_clean)