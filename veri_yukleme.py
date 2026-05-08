import pandas as pd


def veri_yukle(data):
    """
    CSV veya Excel talep verisini okur.

    Zorunlu kolonlar:
    - store
    - date
    - sales

    Türkçe kolon adlarını da otomatik çevirir:
    musteri / müşteri / eczane -> store
    tarih / gün / gun -> date
    talep / satis / satış / demand / miktar -> sales
    """

    if isinstance(data, str):
        file_path = data.lower()

        if file_path.endswith(".csv"):
            df = pd.read_csv(data, low_memory=False)

        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data)

        else:
            raise ValueError("Sadece CSV, XLSX veya XLS dosyası desteklenir.")

    else:
        filename = getattr(data, "filename", "").lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(data, low_memory=False)

        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data)

        else:
            try:
                df = pd.read_csv(data, low_memory=False)
            except Exception:
                df = pd.read_excel(data)

    df.columns = df.columns.astype(str).str.strip().str.lower()

    kolon_eslestirme = {
        "store": "store",
        "store_id": "store",
        "customer": "store",
        "customer_id": "store",
        "musteri": "store",
        "müşteri": "store",
        "musteri_id": "store",
        "müşteri_id": "store",
        "eczane": "store",
        "eczane_id": "store",

        "date": "date",
        "tarih": "date",
        "gun": "date",
        "gün": "date",

        "sales": "sales",
        "sale": "sales",
        "demand": "sales",
        "talep": "sales",
        "satis": "sales",
        "satış": "sales",
        "miktar": "sales",
        "siparis": "sales",
        "sipariş": "sales"
    }

    df = df.rename(columns=kolon_eslestirme)

    gerekli_kolonlar = ["store", "date", "sales"]
    eksik_kolonlar = [col for col in gerekli_kolonlar if col not in df.columns]

    if eksik_kolonlar:
        raise ValueError(
            f"Eksik kolonlar var: {eksik_kolonlar}. "
            "Dosyada store-date-sales kolonları veya Türkçe karşılıkları bulunmalı."
        )

    df = df[gerekli_kolonlar].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["store"] = df["store"].astype(str).str.strip()

    df = df.dropna(subset=["store", "date", "sales"])

    df = df.sort_values(["store", "date"]).reset_index(drop=True)

    return df