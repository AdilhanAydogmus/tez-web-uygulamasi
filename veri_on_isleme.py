import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from veri_yukleme import veri_yukle
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

import joblib 
def veri_on_islem(data, window_size: int = 30, test_ratio: float= 0.10):
    df = veri_yukle(data)

    df["sales_log"] = np.log1p(df["sales"])
    clean_list = []
    for store_id in df["store"].unique():
        
        temp = df[df["store"] == store_id].copy()
        
        z_score = np.abs(zscore(temp["sales_log"]))
        
        temp_clean = temp[z_score < 3]
        clean_list.append(temp_clean)

    df_clean = pd.concat(clean_list)
    df_clean = df_clean.sort_values(["store", "date"])
    scaler = StandardScaler()
    df_clean["sales_scaled"] = scaler.fit_transform(df_clean[["sales_log"]])
    joblib.dump(scaler, "scaler.pkl")
    print("Veri ön işleme tamamlandı.")
    
    

    X = []
    y = []

    for store_id in df_clean["store"].unique():

        temp = df_clean[df_clean["store"] == store_id].sort_values("date")

        values = temp["sales_scaled"].values

        if len(values) <= window_size:
            continue

        for i in range(len(values) - window_size):
            X.append(values[i:i+window_size])
            y.append(values[i+window_size])

    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = np.array(y)
    split_index = int(len(X)*(1-test_ratio))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train,X_test, y_train, y_test, scaler, df_clean

