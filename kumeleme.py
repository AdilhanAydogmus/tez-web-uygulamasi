import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from tensorflow.keras.models import load_model

from veri_on_isleme import veri_on_islem


# =========================
# 1. GELECEK GÜN TAHMİNİ
# =========================

def gelecek_gun_tahmini_yap(model, scaler, store_df, window_size=30):
    values = store_df["sales_scaled"].values

    if len(values) < window_size:
        return 0.0

    input_seq = values[-window_size:].reshape(1, window_size, 1)

    pred_scaled = model.predict(input_seq, verbose=0)[0][0]

    pred_log = scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]

    # Çok büyük tahminleri engelle
    pred_log = np.clip(pred_log, 0, 20)

    pred_real = np.expm1(pred_log)

    if not np.isfinite(pred_real):
        pred_real = 0.0

    return max(0.0, float(pred_real))


# =========================
# 2. MÜŞTERİ PROFİLİ OLUŞTURMA
# =========================

def musteri_profili_olustur(df_clean, model, scaler, window_size=30):

    profil_listesi = []

    for store_id in df_clean["store"].unique():

        store_df = df_clean[df_clean["store"] == store_id].sort_values("date").copy()

        ilk_tarih = store_df["date"].min()
        son_tarih = store_df["date"].max()

        sadakat_suresi_ay = max(
            1,
            (son_tarih.year - ilk_tarih.year) * 12
            + (son_tarih.month - ilk_tarih.month)
            + 1
        )

        toplam_talep = store_df["sales"].sum()
        aylik_ortalama_ciro = toplam_talep / sadakat_suresi_ay

        siparis_gunu = (store_df["sales"] > 0).sum()
        aylik_ortalama_siparis_sikligi = siparis_gunu / sadakat_suresi_ay

        gelecek_gun_tahmini_talep = gelecek_gun_tahmini_yap(
            model=model,
            scaler=scaler,
            store_df=store_df,
            window_size=window_size
        )

        profil_listesi.append({
            "Store": store_id,
            "Sadakat_Suresi_Ay": sadakat_suresi_ay,
            "Aylik_Ortalama_Ciro": aylik_ortalama_ciro,
            "Aylik_Ortalama_Siparis_Sikligi": aylik_ortalama_siparis_sikligi,
            "Gelecek_Gun_Tahmini_Talep": gelecek_gun_tahmini_talep
        })

    return pd.DataFrame(profil_listesi)


# =========================
# 3. KÜMELEME
# =========================

def kumeleme_yap(
    profil_df,
    n_clusters=2,
    output_dir="outputs",
    plot_dir="static/plots"
):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    feature_cols = [
        "Sadakat_Suresi_Ay",
        "Aylik_Ortalama_Ciro",
        "Aylik_Ortalama_Siparis_Sikligi",
        "Gelecek_Gun_Tahmini_Talep"
    ]

    X = profil_df[feature_cols].copy()

    # INF / NAN TEMİZLİĞİ
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # AŞIRI UÇ DEĞERLERİ SINIRLA
    for col in feature_cols:
        upper_limit = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=upper_limit)

    profile_scaler = MinMaxScaler()
    X_scaled = profile_scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    profil_df["Cluster"] = kmeans.fit_predict(X_scaled)

    # =========================
    # SEGMENT SKORU
    # =========================

    profil_df["Segment_Skoru"] = (
        profil_df["Aylik_Ortalama_Ciro"] * 0.40
        + profil_df["Aylik_Ortalama_Siparis_Sikligi"] * 0.30
        + profil_df["Gelecek_Gun_Tahmini_Talep"] * 0.30
    )

    cluster_ozet = profil_df.groupby("Cluster").agg({
        "Store": "count",
        "Sadakat_Suresi_Ay": "mean",
        "Aylik_Ortalama_Ciro": "mean",
        "Aylik_Ortalama_Siparis_Sikligi": "mean",
        "Gelecek_Gun_Tahmini_Talep": "mean",
        "Segment_Skoru": "mean"
    }).reset_index()

    cluster_ozet = cluster_ozet.rename(columns={
        "Store": "Musteri_Sayisi"
    })

    cluster_ozet = cluster_ozet.sort_values(
        "Segment_Skoru",
        ascending=False
    ).reset_index(drop=True)

    segment_map = {}

    for i, row in cluster_ozet.iterrows():
        if i == 0:
            segment_map[row["Cluster"]] = "Altin"
        elif i == 1:
            segment_map[row["Cluster"]] = "Gumus"
        else:
            segment_map[row["Cluster"]] = "Bronz"

    profil_df["Segment"] = profil_df["Cluster"].map(segment_map)

    # =========================
    # SILHOUETTE
    # =========================

    if n_clusters > 1 and len(profil_df) > n_clusters:
        sil_score = silhouette_score(X_scaled, profil_df["Cluster"])
    else:
        sil_score = 0.0

    # =========================
    # PCA GRAFİK
    # =========================

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plot_path = os.path.join(plot_dir, "cluster_plot.png")

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=profil_df["Cluster"],
        cmap="viridis"
    )

    plt.title("KMeans Kümeleme Sonucu")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # =========================
    # MODEL KAYITLARI
    # =========================

    joblib.dump(kmeans, os.path.join(output_dir, "kmeans_model.pkl"))
    joblib.dump(profile_scaler, os.path.join(output_dir, "profile_scaler.pkl"))

    profil_excel_path = os.path.join(output_dir, "musteri_kumeleme_sonuclari.xlsx")
    ozet_excel_path = os.path.join(output_dir, "cluster_ozet.xlsx")

    profil_df.to_excel(profil_excel_path, index=False)
    cluster_ozet.to_excel(ozet_excel_path, index=False)

    return profil_df, cluster_ozet, kmeans, profile_scaler, plot_path, sil_score


# =========================
# 4. ANA PIPELINE
# =========================

def kumeleme_pipeline(
    data_path,
    model_path="lstm_talep_model.h5",
    scaler_path="scaler.pkl",
    window_size=30,
    test_ratio=0.10,
    n_clusters=2,
    output_dir="outputs",
    plot_dir="static/plots"
):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model dosyası bulunamadı: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler dosyası bulunamadı: {scaler_path}")

    model = load_model(model_path, compile=False)

    lstm_scaler = joblib.load(scaler_path)

    X_train, X_test, y_train, y_test, _, df_clean = veri_on_islem(
        data=data_path,
        window_size=window_size,
        test_ratio=test_ratio
    )

    # Önemli:
    # Burada tahmin için yüklenen scaler kullanılıyor.
    profil_df = musteri_profili_olustur(
        df_clean=df_clean,
        model=model,
        scaler=lstm_scaler,
        window_size=window_size
    )

    profil_df, cluster_ozet, kmeans, profile_scaler, plot_path, sil_score = kumeleme_yap(
        profil_df=profil_df,
        n_clusters=n_clusters,
        output_dir=output_dir,
        plot_dir=plot_dir
    )

    return profil_df, cluster_ozet, kmeans, profile_scaler, plot_path, sil_score