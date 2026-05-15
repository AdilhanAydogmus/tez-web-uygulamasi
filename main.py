from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from typing import Annotated, Any, Optional
import os
import uuid
import shutil

from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from sezgisel import VRPData, ALNSSetPartitioning
from kumeleme import kumeleme_pipeline
from lstm_model import lstm_model_egit
from fastapi.responses import (
    HTMLResponse,
    FileResponse,
    JSONResponse
)

app = FastAPI(title="ARAÇ ROTALAMA, LSTM VE MÜŞTERİ SEGMENTASYONU")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)


class RotalamaSonuc(BaseModel):
    dosya_adi: str
    iterasyon: int
    rota: Any
    maliyet: float


@app.get("/", response_class=HTMLResponse, name="home")
async def ana_sayfa(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="rotalama.html"
    )


@app.get("/hakkimizda", response_class=HTMLResponse, name="hakkimizda")
async def hakkimizda_sayfasi(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="hakkimizda.html"
    )


@app.get("/lstm-egitimi", response_class=HTMLResponse, name="lstm_egitimi")
async def lstm_egitimi_sayfasi(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="lstm_egitimi.html"
    )


@app.get("/musteri-segmentasyonu", response_class=HTMLResponse, name="musteri_segmentasyonu")
async def musteri_segmentasyonu_sayfasi(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="musteri_segmentasyonu.html"
    )


# =========================================================
# TEZ VERİLERİ İLE OTOMATİK ROTALAMA
# =========================================================

# =========================================================
# TEZ VERİLERİ İLE HAZIR ROTALAMA SONUCU
# =========================================================

@app.get("/rotalama/tez")
async def rotalama_tez():

    try:

        import pandas as pd
        import ast

        excel_path = (
            "static/rotalama_sonuclari/"
            "matheuristic_sonuc.xlsx"
        )

        df = pd.read_excel(excel_path)

        routes = []

        total_cost = 0

        for _, row in df.iterrows():

            # =====================================================
            # ROTA
            # =====================================================

            route_str = str(row["route"])

            route_list = [

                x.strip()

                for x in route_str.split("->")
            ]

            # =====================================================
            # KOORDİNATLAR
            # =====================================================

            koordinatlar = ast.literal_eval(

                str(row["coordinates"])
            )

            coordinates = []

            for idx, coord in enumerate(koordinatlar):

                node_id = "0"

                if idx < len(route_list):

                    node_id = route_list[idx]

                coordinates.append({

                    "id":
                        str(node_id),

                    "lat":
                        float(coord[0]),

                    "lng":
                        float(coord[1])
                })

            # =====================================================
            # MALİYETLER
            # =====================================================

            route_cost = float(

                str(row["total_cost"])
                .replace(",", ".")
            )

            distance_cost = float(

                str(row["distance_cost"])
                .replace(",", ".")
            )

            late_cost = float(

                str(row["late_cost"])
                .replace(",", ".")
            )

            total_cost += route_cost

            # =====================================================
            # ROUTE APPEND
            # =====================================================

            routes.append({

                "service":
                    str(row["service"]),

                "vehicle":
                    str(row["vehicle"]),

                "route":
                    route_list,

                "coordinates":
                    coordinates,

                "distance_cost":
                    distance_cost,

                "late_cost":
                    late_cost,

                "total_cost":
                    route_cost
            })

        # =====================================================
        # RETURN
        # =====================================================

        return {

            "total_cost":
                total_cost,

            "routes":
                routes,

            "download_excel":
                "/download-rotalama"
        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Tez rotalama sonucu okunamadı: {str(e)}"
        )
# =========================================================
# KENDİ VERİLERİM İLE ROTALAMA
# =========================================================

@app.post(
    "/rotalama/",
    response_model=RotalamaSonuc,
    summary="Araç Rotalarını Optimize Et"
)
async def rotalama_motoru(
    data: Annotated[UploadFile, File()],
    iterations: int = 500
):

    if not data.filename.endswith((".xlsx", ".xls")):

        raise HTTPException(
            status_code=400,
            detail="Lütfen sadece .xlsx veya .xls dosyası yükleyin."
        )

    try:

        file_id = str(uuid.uuid4())

        veri_path = os.path.join(
            "uploads",
            f"{file_id}_{data.filename}"
        )

        with open(veri_path, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)

        segment_path = "static/veriler/musteri_kumeleme_sonuclari.xlsx"

        data_obj = VRPData(
            filepath=veri_path,
            segment_filepath=segment_path
        )

        solver = ALNSSetPartitioning(
            data=data_obj,
            seed=42
        )

        sonuc = solver.solve(iterations=iterations)

        return {
            "dosya_adi": data.filename,
            "iterasyon": iterations,
            "rota": sonuc["routes"],
            "maliyet": float(sonuc["total_cost"]),
            "download_excel": "/download/matheuristic_sonuc.xlsx"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Rotalama sırasında hata oluştu: {str(e)}"
        )


@app.post("/rotalama/kendi-verilerim")
async def rotalama_kendi_verilerim(
    veri_dosyasi: Annotated[UploadFile, File()],
    kume_dosyasi: Optional[UploadFile] = File(default=None),
    iterations: int = Form(default=500)
):

    if not veri_dosyasi.filename.endswith((".xlsx", ".xls")):

        raise HTTPException(
            status_code=400,
            detail="Rotalama veri dosyası Excel olmalıdır."
        )

    try:

        file_id = str(uuid.uuid4())

        veri_path = os.path.join(
            "uploads",
            f"{file_id}_{veri_dosyasi.filename}"
        )

        with open(veri_path, "wb") as buffer:
            shutil.copyfileobj(veri_dosyasi.file, buffer)

        segment_path = "static/veriler/musteri_kumeleme_sonuclari.xlsx"

        if kume_dosyasi is not None and kume_dosyasi.filename:

            segment_path = os.path.join(
                "uploads",
                f"{file_id}_{kume_dosyasi.filename}"
            )

            with open(segment_path, "wb") as buffer:
                shutil.copyfileobj(kume_dosyasi.file, buffer)

        data_obj = VRPData(
            filepath=veri_path,
            segment_filepath=segment_path
        )

        solver = ALNSSetPartitioning(
            data=data_obj,
            seed=42
        )

        sonuc = solver.solve(iterations=iterations)

        return {
            "message": "Rotalama başarıyla tamamlandı.",
            "dosya_adi": veri_dosyasi.filename,
            "iterasyon": iterations,
            "rota": sonuc["routes"],
            "maliyet": float(sonuc["total_cost"]),
            "download_excel":
        "/download/matheuristic_sonuc.xlsx"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Kendi verileriniz ile rotalama sırasında hata oluştu: {str(e)}"
        )


# =========================================================
# SEGMENTASYON
# =========================================================

@app.post("/segmentasyon/tez-verisi")
async def tez_verisi_segmentasyon():

    try:

        import pandas as pd

        profil_df = pd.read_excel(
            "outputs/musteri_kumeleme_sonuclari.xlsx"
        )

        cluster_ozet = pd.read_excel(
            "outputs/cluster_ozet.xlsx"
        )

        return {

            "success": True,

            "customers":
                profil_df.to_dict(orient="records"),

            "cluster_summary":
                cluster_ozet.to_dict(orient="records"),

            "plot_url":
                "/static/kumeleme_sonuclari/cluster_plot.png",

            "silhouette_plot_url":
                "/static/kumeleme_sonuclari/silhouette_plot.png",

            "silhouette_score":
                0.61,

            "best_k":
                2,

            "excel_1":
                "/download/musteri_kumeleme_sonuclari.xlsx",

            "excel_2":
                "/download/cluster_ozet.xlsx"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Tez segmentasyon verisi okunamadı: {str(e)}"
        )

@app.post("/segmentasyon/kendi-verim")
async def kendi_verim_ile_segmentasyon(
    data: Annotated[UploadFile, File()],
    model_file: Optional[UploadFile] = File(default=None),
    scaler_file: Optional[UploadFile] = File(default=None),
    n_clusters: int = Form(default=3),
    window_size: int = Form(default=30)
):

    try:

        if not data.filename.endswith(
            (".csv", ".xlsx", ".xls")
        ):

            raise HTTPException(
                status_code=400,
                detail="Lütfen CSV veya Excel dosyası yükleyin."
            )

        file_id = str(uuid.uuid4())

        upload_path = os.path.join(
            "uploads",
            f"{file_id}_{data.filename}"
        )

        with open(upload_path, "wb") as buffer:

            shutil.copyfileobj(
                data.file,
                buffer
            )

        model_path = "lstm_talep_model.h5"

        scaler_path = "scaler.pkl"

        if model_file is not None and model_file.filename:

            model_path = os.path.join(
                "uploads",
                f"{file_id}_{model_file.filename}"
            )

            with open(model_path, "wb") as buffer:

                shutil.copyfileobj(
                    model_file.file,
                    buffer
                )

        if scaler_file is not None and scaler_file.filename:

            scaler_path = os.path.join(
                "uploads",
                f"{file_id}_{scaler_file.filename}"
            )

            with open(scaler_path, "wb") as buffer:

                shutil.copyfileobj(
                    scaler_file.file,
                    buffer
                )

        (
            profil_df,
            cluster_ozet,
            kmeans,
            profile_scaler,
            plot_path,
            silhouette_plot_path,
            sil_score,
            best_k

        ) = kumeleme_pipeline(
            data_path=upload_path,
            model_path=model_path,
            scaler_path=scaler_path,
            window_size=window_size,
            test_ratio=0.10,
            n_clusters=n_clusters,
            output_dir="outputs",
            plot_dir="static/plots"
        )

        return {

            "message":
                "Yüklenen veri ile kümeleme tamamlandı.",

            "silhouette_score":
                float(sil_score),

            "best_k":
                int(best_k),

            "plot_url":
                "/" + plot_path.replace("\\", "/"),

            "silhouette_plot_url":
                "/" + silhouette_plot_path.replace("\\", "/"),

            "excel_1":
                "/download/musteri_kumeleme_sonuclari.xlsx",

            "excel_2":
                "/download/cluster_ozet.xlsx",

            "customers":
                profil_df.head(100).to_dict(
                    orient="records"
                )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Yüklenen veri ile kümeleme sırasında hata oluştu: {str(e)}"
        )


# =========================================================
# LSTM
# =========================================================

@app.post("/segmentasyon/lstm-egit")
async def lstm_egit_endpoint(
    data: Annotated[UploadFile, File()],
    epochs: int = Form(default=20),
    window_size: int = Form(default=30)
):

    try:

        if not data.filename.endswith((".csv", ".xlsx", ".xls")):

            raise HTTPException(
                status_code=400,
                detail="Lütfen CSV veya Excel dosyası yükleyin."
            )

        file_id = str(uuid.uuid4())

        upload_path = os.path.join(
            "uploads",
            f"{file_id}_{data.filename}"
        )

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)

        sonuc = lstm_model_egit(
            data=upload_path,
            window_size=window_size,
            epochs=epochs,
            output_dir="outputs",
            plot_dir="static/plots"
        )

        return {

    "mae":
        sonuc["mae"],

    "rmse":
        sonuc["rmse"],

    "mape":
        sonuc["mape"],

    "epoch_count":
        sonuc["epoch_count"],

    "final_loss":
        sonuc["final_loss"],

    "final_val_loss":
        sonuc["final_val_loss"],

    "loss_plot_url":
        "/" + sonuc["loss_plot"].replace("\\", "/"),

    "prediction_plot_url":
        "/" + sonuc["prediction_plot"].replace("\\", "/"),

    "model_download":
        "/download/lstm_talep_model.h5",

    "scaler_download":
        "/download/scaler.pkl"
}

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"LSTM eğitimi sırasında hata oluştu: {str(e)}"
        )


@app.get("/download/{filename}")
async def download_file(filename: str):

    file_path = os.path.join("outputs", filename)

    if not os.path.exists(file_path):

        raise HTTPException(
            status_code=404,
            detail="Dosya bulunamadı."
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )
# =========================================================
# HAZIR ROTALAMA EXCEL İNDİRME
# =========================================================

@app.get("/download-rotalama")
async def download_rotalama():

    path = (
        "static/rotalama_sonuclari/"
        "matheuristic_sonuc.xlsx"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Rotalama sonucu bulunamadı."
        )

    return FileResponse(
        path=path,
        filename="matheuristic_sonuc.xlsx",
        media_type="application/octet-stream"
    )
