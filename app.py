"""STT Italiano — web app locale (FastAPI + Whisper, 100% on-device).

Backend automatico (vedi stt_engine.py): mlx-whisper su Mac Apple Silicon,
faster-whisper su Windows / Linux / Mac Intel.

Avvio:  uvicorn app:app --host 127.0.0.1 --port 9753
Poi apri http://localhost:9753
"""

import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import stt_engine

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {
    ".ogg", ".wav", ".mp3", ".mp4", ".m4a", ".webm",
    ".flac", ".aac", ".opus", ".mov", ".mkv",
}

ALLOWED_LANGUAGES = {"it", "en", "es", "fr", "de", "auto"}

# Limite upload: 2 GB ≈ 3 ore di WAV non compresso (ogg/mp3 sono 10-30x più piccoli).
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024

# Quanti job restano in memoria: i più vecchi vengono eliminati (LRU).
MAX_JOBS = 100

# Numero massimo di file caricabili in un'unica richiesta batch.
MAX_BATCH_FILES = 10

# Pulizia degli upload rimasti da eventuali crash precedenti.
for _f in UPLOAD_DIR.iterdir():
    if _f.is_file():
        _f.unlink(missing_ok=True)

app = FastAPI(title="STT Italiano — Whisper locale")

# --- Stato in memoria (app locale single-user) ---
_jobs: dict = {}
_jobs_lock = threading.Lock()
_downloads: dict = {}  # model_id -> {"status": ..., "error": ...}
_downloads_lock = threading.Lock()

_transcribe_pool = ThreadPoolExecutor(max_workers=1)  # una trascrizione alla volta
_download_pool = ThreadPoolExecutor(max_workers=2)


# ------------------------------- Modelli -------------------------------

@app.get("/api/health")
def health():
    """Endpoint leggerissimo per il heartbeat del frontend (rilevamento motore offline)."""
    return {"status": "ok"}


@app.get("/api/models")
def list_models():
    out = []
    for mid, meta in stt_engine.MODELS.items():
        if not stt_engine.is_supported(mid):
            continue  # modello nascosto: non compatibile con questa piattaforma
        with _downloads_lock:
            dl = _downloads.get(mid, {})
        out.append(
            {
                "id": mid,
                **meta,
                "default": mid == stt_engine.DEFAULT_MODEL,
                "downloaded": stt_engine.is_downloaded(mid),
                "downloading": dl.get("status") == "downloading",
                "download_error": dl.get("error"),
            }
        )
    return {"models": out, "backend": stt_engine.BACKEND}


def _download_worker(model_id: str):
    try:
        stt_engine.download_model(model_id)
        with _downloads_lock:
            _downloads[model_id] = {"status": "done", "error": None}
    except Exception as exc:  # noqa: BLE001
        with _downloads_lock:
            _downloads[model_id] = {"status": "error", "error": str(exc)}


@app.post("/api/models/{model_id}/download")
def download_model(model_id: str):
    if model_id not in stt_engine.MODELS:
        raise HTTPException(404, "Modello sconosciuto")
    if not stt_engine.is_supported(model_id):
        raise HTTPException(400, "Modello non disponibile su questa piattaforma")
    if stt_engine.is_downloaded(model_id):
        return {"status": "downloaded"}
    with _downloads_lock:
        if _downloads.get(model_id, {}).get("status") == "downloading":
            return {"status": "downloading"}
        _downloads[model_id] = {"status": "downloading", "error": None}
    _download_pool.submit(_download_worker, model_id)
    return {"status": "downloading"}


# ----------------------------- Trascrizioni -----------------------------

def _transcribe_worker(job_id: str, audio_path: Path, model_ids: list[str], language: str):
    """Esegue i modelli in sequenza (cascata) sullo stesso file audio.

    Un modello fallito non blocca gli altri: l'errore viene registrato nella
    sua sezione di `results`. Il job è "error" solo se falliscono TUTTI i modelli.
    """
    with _jobs_lock:
        _jobs[job_id].update(status="running", started_at=time.time())
    try:
        for i, model_id in enumerate(model_ids, 1):
            with _jobs_lock:
                _jobs[job_id].update(current_model=model_id, current_index=i)
            t0 = time.time()
            try:
                result = stt_engine.transcribe(str(audio_path), model_id, language)
                result["elapsed"] = round(time.time() - t0, 1)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc), "elapsed": round(time.time() - t0, 1)}
            with _jobs_lock:
                _jobs[job_id]["results"][model_id] = result
        with _jobs_lock:
            results = _jobs[job_id]["results"]
            errors = [r["error"] for r in results.values() if r.get("error")]
            if len(errors) == len(model_ids):
                _jobs[job_id].update(
                    status="error", error=" | ".join(errors), finished_at=time.time()
                )
            else:
                _jobs[job_id].update(status="done", finished_at=time.time())
            _jobs[job_id].update(current_model=None)
    finally:
        audio_path.unlink(missing_ok=True)  # l'upload non serve più


def _validate_models_and_language(models: list[str], language: str) -> list[str]:
    """Valida modelli e lingua; ritorna la lista deduplicata dei model_id."""
    model_ids = list(dict.fromkeys(models))  # dedup preservando l'ordine
    if not model_ids:
        raise HTTPException(400, "Seleziona almeno un modello")
    for model_id in model_ids:
        if model_id not in stt_engine.MODELS:
            raise HTTPException(400, f"Modello sconosciuto: {model_id}")
        if not stt_engine.is_supported(model_id):
            raise HTTPException(
                400, f"Modello non disponibile su questa piattaforma: {model_id}"
            )
        if not stt_engine.is_downloaded(model_id):
            raise HTTPException(400, f"Modello non ancora scaricato: {model_id}")
    if language not in ALLOWED_LANGUAGES:
        raise HTTPException(400, f"Lingua non supportata: {language}")
    return model_ids


async def _save_upload(file: UploadFile) -> tuple[str, Path]:
    """Salva l'upload su disco in streaming; ritorna (job_id, audio_path)."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Formato non supportato: {ext or '(nessuna estensione)'}")
    job_id = uuid.uuid4().hex[:12]
    audio_path = UPLOAD_DIR / f"{job_id}{ext}"
    try:
        size = 0
        with audio_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):  # 1 MB alla volta, niente full-read in RAM
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"File troppo grande: max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                        " (≈3 ore di audio)",
                    )
                out.write(chunk)
    except Exception:
        audio_path.unlink(missing_ok=True)
        raise
    return job_id, audio_path


def _register_job(
    job_id: str, filename: str | None, model_ids: list[str], language: str, audio_path: Path
):
    """Registra il job in memoria e lo mette in coda per la trascrizione."""
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "filename": filename,
            "models": model_ids,
            "language": language,
            "created_at": time.time(),
            "results": {},       # model_id -> trascrizione (+elapsed) oppure {"error": ...}
            "current_model": None,
            "current_index": 0,
            "error": None,
        }
        while len(_jobs) > MAX_JOBS:
            _jobs.pop(next(iter(_jobs)))  # i dict sono ordinati: elimina il più vecchio
    _transcribe_pool.submit(_transcribe_worker, job_id, audio_path, model_ids, language)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    models: list[str] = Form(...),
    language: str = Form("it"),
):
    # Uno o più modelli, eseguiti in cascata nell'ordine di selezione.
    model_ids = _validate_models_and_language(models, language)
    job_id, audio_path = await _save_upload(file)
    _register_job(job_id, file.filename, model_ids, language, audio_path)
    return {"job_id": job_id}


@app.post("/api/transcribe-batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    models: list[str] = Form(...),
    language: str = Form("it"),
):
    """Upload multiplo: fino a MAX_BATCH_FILES file, ognuno diventa un job separato."""
    if not files:
        raise HTTPException(400, "Nessun file caricato")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f"Massimo {MAX_BATCH_FILES} file per volta")
    model_ids = _validate_models_and_language(models, language)
    # Valida tutte le estensioni prima di salvare qualsiasi file.
    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(
                400,
                f"Formato non supportato: {ext or '(nessuna estensione)'} ({file.filename})",
            )
    jobs_out = []
    for file in files:
        job_id, audio_path = await _save_upload(file)
        _register_job(job_id, file.filename, model_ids, language, audio_path)
        jobs_out.append({"job_id": job_id, "filename": file.filename})
    return {"jobs": jobs_out}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job non trovato")
        data = dict(job)
    now = time.time()
    end = data.pop("finished_at", None) or now
    data["elapsed"] = round(end - data.get("started_at", data["created_at"]), 1)
    data.pop("started_at", None)
    data.pop("created_at", None)
    return data


@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str, fmt: str = "txt", model: str | None = None):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(400, "Trascrizione non completata")
    stem = _safe_filename(job["filename"])
    if fmt == "combined":
        # Un unico file con tutte le trascrizioni (sezioni per modello).
        ordered = [(mid, job["results"][mid]) for mid in job["models"] if mid in job["results"]]
        return PlainTextResponse(
            stt_engine.to_multi_txt(ordered, job["filename"]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}.multi.txt"'},
        )
    # Export del singolo modello (default: il primo del job).
    model_id = model or job["models"][0]
    result = job["results"].get(model_id)
    if result is None or result.get("error"):
        raise HTTPException(404, f"Trascrizione non disponibile per il modello: {model_id}")
    tag = f".{model_id}" if len(job["models"]) > 1 else ""
    if fmt == "srt":
        return PlainTextResponse(
            stt_engine.to_srt(result["segments"]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}{tag}.srt"'},
        )
    return PlainTextResponse(
        result["text"],
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}{tag}.txt"'},
    )


def _safe_filename(name: str) -> str:
    """Nome file sicuro per l'header Content-Disposition (niente quote, CRLF o path)."""
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", Path(name or "").stem).strip()
    return (stem or "trascrizione")[:80]


# ------------------------------- Frontend -------------------------------

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")