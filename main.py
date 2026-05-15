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

@app.get("/rotalama/tez")
async def rotalama_tez():

    try:

        veri_path = "static/veriler/vrp_data.xlsx"
        segment_path = "static/veriler/musteri_kumeleme_sonuclari.xlsx"

        data = VRPData(
            filepath=veri_path,
            segment_filepath=segment_path
        )

        solver = ALNSSetPartitioning(
            data=data,
            seed=42
        )

        sonuc = solver.solve(iterations=2000)

        return sonuc

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Tez verisi rotalama hatası: {str(e)}"
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
            "maliyet": float(sonuc["total_cost"])
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
            "maliyet": float(sonuc["total_cost"])
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

            data_path="talepverisitez.xlsx",

            model_path="lstm_talep_model.h5",

            scaler_path="scaler.pkl",

            window_size=30,

            n_clusters=3
        )

        return {

            "success": True,

            "customers":
                profil_df.to_dict(orient="records"),

            "cluster_summary":
                cluster_ozet.to_dict(orient="records"),

            "plot_url":
                "/" + plot_path.replace("\\", "/"),

            "silhouette_plot_url":
                "/" + silhouette_plot_path.replace("\\", "/"),

            "silhouette_score":
                float(sil_score),

            "best_k":
                int(best_k),

            "excel_1":
                "/outputs/musteri_kumeleme_sonuclari.xlsx",

            "excel_2":
                "/outputs/cluster_ozet.xlsx"
        }
        

    except Exception as e:

        import traceback

        traceback.print_exc()

        return JSONResponse(

            status_code=500,

            content={

                "detail": str(e)
            }
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

        return sonuc

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