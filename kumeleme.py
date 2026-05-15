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


# =====================================================
# 1. GELECEK GÜN TAHMİNİ
# =====================================================

def gelecek_gun_tahmini_yap(
    model,
    store_df,
    window_size=30
):

    # =================================================
    # SALES
    # =================================================

    values = store_df["sales"].values.astype(float)

    # =================================================
    # YETERLİ VERİ KONTROLÜ
    # =================================================

    if len(values) < window_size:

        return 0.0

    # =================================================
    # NORMALİZASYON
    # =================================================

    local_scaler = MinMaxScaler()

    values_scaled = local_scaler.fit_transform(
        values.reshape(-1, 1)
    )

    # =================================================
    # MODEL INPUT
    # =================================================

    input_seq = values_scaled[-window_size:].reshape(
        1,
        window_size,
        1
    )

    # =================================================
    # MODEL TAHMİNİ
    # =================================================

    pred_scaled = model.predict(
        input_seq,
        verbose=0
    )[0][0]

    # =================================================
    # SCALE GERİ ÇEVİR
    # =================================================

    pred_real = local_scaler.inverse_transform(
        np.array([[pred_scaled]])
    )[0][0]

    # =================================================
    # NAN / INF
    # =================================================

    if not np.isfinite(pred_real):

        pred_real = 0.0

    # =================================================
    # NEGATİF ENGELİ
    # =================================================

    pred_real = max(
        0.0,
        float(pred_real)
    )

    return pred_real


# =====================================================
# 2. MÜŞTERİ PROFİLİ
# =====================================================

def musteri_profili_olustur(
    df_clean,
    model,
    window_size=30
):

    profil_listesi = []

    # =================================================
    # HER STORE
    # =================================================

    for store_id in df_clean["store"].unique():

        store_df = (

            df_clean[
                df_clean["store"] == store_id
            ]

            .sort_values("date")

            .copy()
        )

        # =================================================
        # TARİHLER
        # =================================================

        ilk_tarih = store_df["date"].min()

        son_tarih = store_df["date"].max()

        # =================================================
        # SADAKAT
        # =================================================

        sadakat_suresi_ay = max(

            1,

            (son_tarih.year - ilk_tarih.year) * 12

            +

            (son_tarih.month - ilk_tarih.month)

            + 1
        )

        # =================================================
        # TOPLAM CİRO
        # =================================================

        toplam_ciro = (

            store_df["sales"].sum()
        )

        # =================================================
        # AYLIK ORTALAMA CİRO
        # =================================================

        aylik_ortalama_ciro = (

            toplam_ciro / sadakat_suresi_ay
        )

        # =================================================
        # SİPARİŞ GÜNÜ
        # =================================================

        siparis_gunu = (

            store_df["sales"] > 0
        ).sum()

        # =================================================
        # AYLIK SİPARİŞ SIKLIĞI
        # =================================================

        aylik_ortalama_siparis_sikligi = (

            siparis_gunu / sadakat_suresi_ay
        )

        # =================================================
        # LSTM TAHMİNİ
        # =================================================

        gelecek_gun_tahmini_talep = (

            gelecek_gun_tahmini_yap(

                model=model,

                store_df=store_df,

                window_size=window_size
            )
        )

        # =================================================
        # PROFİL
        # =================================================

        profil_listesi.append({

            "Store":
                int(store_id),

            "Sadakat_Suresi_Ay":
                float(sadakat_suresi_ay),

            "Aylik_Ortalama_Ciro":
                float(aylik_ortalama_ciro),

            "Aylik_Ortalama_Siparis_Sikligi":
                float(aylik_ortalama_siparis_sikligi),

            "Gelecek_Gun_Tahmini_Talep":
                float(gelecek_gun_tahmini_talep)
        })

    # =================================================
    # DATAFRAME
    # =================================================

    profil_df = pd.DataFrame(
        profil_listesi
    )

    return profil_df


# =====================================================
# 3. KÜMELEME
# =====================================================

def kumeleme_yap(
    profil_df,
    n_clusters=3,
    output_dir="outputs",
    plot_dir="static/plots"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    # =================================================
    # FEATURELAR
    # =================================================

    feature_cols = [

        "Sadakat_Suresi_Ay",

        "Aylik_Ortalama_Ciro",

        "Aylik_Ortalama_Siparis_Sikligi",

        "Gelecek_Gun_Tahmini_Talep"
    ]

    # =================================================
    # X
    # =================================================

    X = profil_df[
        feature_cols
    ].copy()

    # =================================================
    # NUMERIC
    # =================================================

    for col in feature_cols:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # =================================================
    # NAN / INF
    # =================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # =================================================
    # NEGATİF
    # =================================================

    for col in feature_cols:

        X[col] = X[col].clip(lower=0)

    # =================================================
    # NORMALİZASYON
    # =================================================

    profile_scaler = MinMaxScaler()

    X_scaled = profile_scaler.fit_transform(X)

    # =================================================
    # KMEANS
    # =================================================

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    profil_df["Cluster"] = (

        kmeans.fit_predict(
            X_scaled
        )
    )

    # =================================================
    # SEGMENT SKORU
    # =================================================

    profil_df["Segment_Skoru"] = (

        profil_df["Aylik_Ortalama_Ciro"] * 0.40

        +

        profil_df["Aylik_Ortalama_Siparis_Sikligi"] * 0.30

        +

        profil_df["Gelecek_Gun_Tahmini_Talep"] * 0.30
    )

    # =================================================
    # CLUSTER ÖZET
    # =================================================

    cluster_ozet = (

        profil_df

        .groupby("Cluster")

        .agg({

            "Store": "count",

            "Sadakat_Suresi_Ay": "mean",

            "Aylik_Ortalama_Ciro": "mean",

            "Aylik_Ortalama_Siparis_Sikligi": "mean",

            "Gelecek_Gun_Tahmini_Talep": "mean",

            "Segment_Skoru": "mean"
        })

        .reset_index()
    )

    # =================================================
    # RENAME
    # =================================================

    cluster_ozet = cluster_ozet.rename(

        columns={

            "Store":
                "Musteri_Sayisi"
        }
    )

    # =================================================
    # SCORE SIRALAMA
    # =================================================

    cluster_ozet = cluster_ozet.sort_values(

        "Segment_Skoru",

        ascending=False

    ).reset_index(drop=True)

    # =================================================
    # SEGMENT MAP
    # =================================================

    segment_map = {}

    for i, row in cluster_ozet.iterrows():

        if i == 0:

            segment_map[
                row["Cluster"]
            ] = "Altin"

        elif i == 1:

            segment_map[
                row["Cluster"]
            ] = "Gumus"

        else:

            segment_map[
                row["Cluster"]
            ] = "Bronz"

    profil_df["Segment"] = (

        profil_df["Cluster"]

        .map(segment_map)
    )

    # =================================================
    # SILHOUETTE
    # =================================================

    try:

        sil_score = silhouette_score(
            X_scaled,
            profil_df["Cluster"]
        )

        if np.isnan(sil_score):

            sil_score = 0.0

    except:

        sil_score = 0.0

    # =================================================
    # PCA
    # =================================================

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    # =================================================
    # GRAFİK
    # =================================================

    plot_path = os.path.join(
        plot_dir,
        "cluster_plot.png"
    )

    plt.figure(figsize=(9, 6))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=profil_df["Cluster"],
        cmap="viridis"
    )

    plt.title(
        "KMeans Kümeleme Sonucu"
    )

    plt.xlabel("PCA 1")

    plt.ylabel("PCA 2")

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_path)

    plt.close()

    # =================================================
    # EXCEL
    # =================================================

    profil_df = profil_df.sort_values(
        "Store"
    )

    profil_excel_path = os.path.join(
        output_dir,
        "musteri_kumeleme_sonuclari.xlsx"
    )

    ozet_excel_path = os.path.join(
        output_dir,
        "cluster_ozet.xlsx"
    )

    profil_df.to_excel(
        profil_excel_path,
        index=False
    )

    cluster_ozet.to_excel(
        ozet_excel_path,
        index=False
    )

    # =================================================
    # MODEL SAVE
    # =================================================

    joblib.dump(
        kmeans,
        os.path.join(
            output_dir,
            "kmeans_model.pkl"
        )
    )

    joblib.dump(
        profile_scaler,
        os.path.join(
            output_dir,
            "profile_scaler.pkl"
        )
    )

    return (
        profil_df,
        cluster_ozet,
        kmeans,
        profile_scaler,
        plot_path,
        sil_score
    )


# =====================================================
# 4. SILHOUETTE ANALİZİ
# =====================================================

def silhouette_grafigi_ciz(
    profil_df,
    max_k=10,
    plot_dir="static/plots"
):

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    feature_cols = [

        "Sadakat_Suresi_Ay",

        "Aylik_Ortalama_Ciro",

        "Aylik_Ortalama_Siparis_Sikligi",

        "Gelecek_Gun_Tahmini_Talep"
    ]

    X = profil_df[
        feature_cols
    ].copy()

    for col in feature_cols:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    scores = []

    max_possible_k = min(
        max_k,
        len(X_scaled) - 1
    )

    k_values = range(
        2,
        max_possible_k + 1
    )

    for k in k_values:

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = km.fit_predict(
            X_scaled
        )

        try:

            score = silhouette_score(
                X_scaled,
                labels
            )

        except:

            score = 0

        scores.append(score)
        scores = np.nan_to_num(scores, nan=0.0)

    if len(scores) == 0:

        return (
            os.path.join(plot_dir, "silhouette_plot.png"),
            3,
            0.0
        )

    best_index = np.argmax(scores)

    best_k = list(k_values)[
        best_index
    ]

    best_score = scores[
        best_index
    ]

    # =================================================
    # GRAFİK
    # =================================================

    plot_path = os.path.join(
        plot_dir,
        "silhouette_plot.png"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        scores,
        marker="o"
    )

    plt.title(
        "Silhouette Skoruna Göre Optimal Küme Sayısı"
    )

    plt.xlabel(
        "Küme Sayısı (k)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.xticks(
        list(k_values)
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_path)

    plt.close()

    return (
        plot_path,
        best_k,
        best_score
    )


# =====================================================
# 5. PIPELINE
# =====================================================

def kumeleme_pipeline(
    data_path,
    model_path="lstm_talep_model.h5",
    scaler_path="scaler.pkl",
    window_size=30,
    test_ratio=0.10,
    n_clusters=3,
    output_dir="outputs",
    plot_dir="static/plots"
):

    # =================================================
    # MODEL
    # =================================================

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}"
        )

    model = load_model(
        model_path,
        compile=False
    )

    # =================================================
    # VERİ ÖN İŞLEME
    # =================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        df_clean

    ) = veri_on_islem(

        data=data_path,

        window_size=window_size,

        test_ratio=test_ratio
    )

    # =================================================
    # PROFİL
    # =================================================

    profil_df = musteri_profili_olustur(

        df_clean=df_clean,

        model=model,

        window_size=window_size
    )

    # =================================================
    # SILHOUETTE
    # =================================================

    (
        silhouette_plot_path,
        best_k,
        best_score

    ) = silhouette_grafigi_ciz(

        profil_df=profil_df,

        max_k=10,

        plot_dir=plot_dir
    )

    # =================================================
    # KÜMELEME
    # =================================================

    (
        profil_df,
        cluster_ozet,
        kmeans,
        profile_scaler,
        plot_path,
        sil_score

    ) = kumeleme_yap(

        profil_df=profil_df,

        n_clusters=best_k,

        output_dir=output_dir,

        plot_dir=plot_dir
    )

    sil_score = best_score

    return (

        profil_df,

        cluster_ozet,

        kmeans,

        profile_scaler,

        plot_path,

        silhouette_plot_path,

        sil_score,

        best_k
    )
