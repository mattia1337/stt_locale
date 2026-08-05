"""Motore STT: registry modelli, download da HuggingFace e trascrizione.

Backend selezionato automaticamente in base alla piattaforma:

- **mlx-whisper** su macOS Apple Silicon: usa il framework MLX di Apple -> GPU Metal.
- **faster-whisper** (CTranslate2) su Windows / Linux / Mac Intel: CPU (int8) di default,
  GPU NVIDIA (CUDA) se disponibile. È il backend cross-platform di riferimento.
- **parakeet-mlx** (solo Apple Silicon): NVIDIA Parakeet TDT 0.6B v3, multilingue (IT),
  velocissimo. I modelli non compatibili con la piattaforma sono nascosti dall'API.

I modelli vengono scaricati automaticamente nella cache di HuggingFace
(macOS/Linux: ~/.cache/huggingface — Windows: %USERPROFILE%\\.cache\\huggingface).
La decodifica audio (mp3/ogg/mp4/... -> PCM 16kHz) avviene in-process via PyAV:
le librerie ffmpeg sono incluse nella wheel pip, niente da installare a livello di sistema.
"""

import importlib.util
import platform
import sys
import threading

from huggingface_hub import snapshot_download

# Registry dei modelli disponibili.
#   mlx_repo: repo HuggingFace in formato MLX (usato da mlx-whisper)
#   fw_model: nome modello risolto da faster-whisper (repo CT2 gestita da faster-whisper)
MODELS = {
    "large-v3-turbo": {
        "mlx_repo": "mlx-community/whisper-large-v3-turbo",
        "fw_model": "large-v3-turbo",
        "label": "Large v3 Turbo",
        "params": "809M",
        "ram": "~2 GB",
        "speed": "Molto veloce",
        "quality": 4,
        "desc": "Consigliato: italiano quasi identico a large-v3, ma ~8x più veloce.",
    },
    "large-v3": {
        "mlx_repo": "mlx-community/whisper-large-v3-mlx",
        "fw_model": "large-v3",
        "label": "Large v3",
        "params": "1.55B",
        "ram": "~4 GB",
        "speed": "Veloce",
        "quality": 5,
        "desc": "Precisione massima: audio rumorosi, accenti marcati, parlato sovrapposto.",
    },
    "medium": {
        "mlx_repo": "mlx-community/whisper-medium-mlx",
        "fw_model": "medium",
        "label": "Medium",
        "params": "769M",
        "ram": "~2 GB",
        "speed": "Molto veloce",
        "quality": 3,
        "desc": "Buon compromesso per bozze di qualità su audio pulito.",
    },
    "small": {
        "mlx_repo": "mlx-community/whisper-small-mlx",
        "fw_model": "small",
        "label": "Small",
        "params": "244M",
        "ram": "~1 GB",
        "speed": "Rapidissimo",
        "quality": 2,
        "desc": "Per note vocali e bozze veloci; qualità italiano discreta.",
    },
    "tiny": {
        "mlx_repo": "mlx-community/whisper-tiny",
        "fw_model": "tiny",
        "label": "Tiny",
        "params": "39M",
        "ram": "<1 GB",
        "speed": "Istantaneo",
        "quality": 1,
        "desc": "Solo per test rapidi; qualità italiano debole.",
    },
    # --- Famiglia Parakeet (NVIDIA): engine diverso, NON è Whisper ---
    "parakeet-v3": {
        "engine": "parakeet",  # i modelli senza questa chiave sono "whisper"
        "mlx_repo": "mlx-community/parakeet-tdt-0.6b-v3",
        "label": "Parakeet v3",
        "params": "600M",
        "ram": "~1.5 GB",
        "speed": "Fulmineo",
        "quality": 4,
        "desc": "Novità: Parakeet (di NVIDIA) in formato MLX: gira sulla GPU Apple, "
                "niente CUDA. Multilingue (italiano ✓), velocissimo e ottimo su parlato "
                "pulito; su audio difficili large-v3 resta superiore.",
    },
}


def _detect_backend() -> str | None:
    """Sceglie il backend STT in base a cosa è installato (senza importare le librerie)."""
    if importlib.util.find_spec("mlx_whisper") is not None:
        return "mlx"
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster-whisper"
    return None


# Backend attivo: "mlx" su Apple Silicon, "faster-whisper" su Windows/Linux/Mac Intel.
BACKEND = _detect_backend()

IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def is_supported(model_id: str) -> bool:
    """True se il modello può girare su questa piattaforma.

    I modelli Parakeet richiedono parakeet-mlx -> solo Apple Silicon.
    """
    meta = MODELS.get(model_id)
    if meta is None:
        return False
    if meta.get("engine", "whisper") == "parakeet":
        return IS_APPLE_SILICON and importlib.util.find_spec("parakeet_mlx") is not None
    return BACKEND is not None


# Default: Parakeet v3 su Apple Silicon (miglior equilibrio fedeltà/leggibilità
# dai nostri test comparativi); large-v3-turbo dove Parakeet non è supportato
# (Windows/Linux/Mac Intel).
DEFAULT_MODEL = "parakeet-v3" if is_supported("parakeet-v3") else "large-v3-turbo"

_BACKEND_HINT = (
    "Nessun backend STT installato. Su Mac Apple Silicon: `pip install mlx-whisper`. "
    "Su Windows/Linux: `pip install faster-whisper` (già incluso in requirements.txt)."
)


# ----------------------------- Download modelli -----------------------------

def is_downloaded(model_id: str) -> bool:
    """True se il modello è già presente nella cache locale di HuggingFace."""
    if not is_supported(model_id):
        return False
    if MODELS[model_id].get("engine", "whisper") == "parakeet" or BACKEND == "mlx":
        try:
            snapshot_download(MODELS[model_id]["mlx_repo"], local_files_only=True)
            return True
        except Exception:
            return False
    if BACKEND == "faster-whisper":
        try:
            # download_model di faster-whisper risolve il nome -> repo CT2 e
            # con local_files_only=True solleva un'eccezione se non in cache.
            from faster_whisper.utils import download_model as fw_download

            fw_download(MODELS[model_id]["fw_model"], local_files_only=True)
            return True
        except Exception:
            return False
    return False


def download_model(model_id: str) -> str:
    """Scarica il modello da HuggingFace (idempotente). Ritorna il path locale."""
    if model_id not in MODELS:
        raise ValueError(f"Modello sconosciuto: {model_id}")
    if not is_supported(model_id):
        raise ValueError(f"Modello non disponibile su questa piattaforma: {model_id}")
    if MODELS[model_id].get("engine", "whisper") == "parakeet" or BACKEND == "mlx":
        return snapshot_download(MODELS[model_id]["mlx_repo"])
    if BACKEND == "faster-whisper":
        from faster_whisper.utils import download_model as fw_download

        return str(fw_download(MODELS[model_id]["fw_model"]))
    raise RuntimeError(_BACKEND_HINT)


# ----------------------------- Decodifica audio -----------------------------

def load_audio(path: str, sr: int = 16000):
    """Decodifica qualsiasi formato (mp3/ogg/mp4/m4a/...) in PCM float32 mono 16kHz.

    Usa PyAV (librerie ffmpeg incluse nella wheel): nessun binario di sistema richiesto.
    """
    import av
    import numpy as np

    container = av.open(str(path))
    if not container.streams.audio:
        container.close()
        raise ValueError("Il file non contiene una traccia audio")
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)

    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray().reshape(-1))
    for resampled in resampler.resample(None):  # flush finale del resampler
        chunks.append(resampled.to_ndarray().reshape(-1))
    container.close()

    if not chunks:
        raise ValueError("Traccia audio vuota o non decodificabile")
    return np.concatenate(chunks).astype("float32")


# ----------------------------- Trascrizione -----------------------------

def transcribe(audio_path: str, model_id: str, language: str | None = "it") -> dict:
    """Trascrive un file audio/video con il modello scelto.

    Ritorna: {"text", "language", "segments": [{"start", "end", "text"}]}
    """
    if model_id not in MODELS:
        raise ValueError(f"Modello sconosciuto: {model_id}")

    if MODELS[model_id].get("engine", "whisper") == "parakeet":
        return _transcribe_parakeet(audio_path, model_id)

    lang = None if language in (None, "", "auto") else language
    if BACKEND == "mlx":
        return _transcribe_mlx(audio_path, model_id, lang)
    if BACKEND == "faster-whisper":
        return _transcribe_faster_whisper(audio_path, model_id, lang)
    raise RuntimeError(_BACKEND_HINT)


def _transcribe_mlx(audio_path: str, model_id: str, lang: str | None) -> dict:
    """Backend Apple Silicon: mlx-whisper su GPU Metal."""
    import mlx_whisper  # import lazy: evita di caricare MLX all'avvio del server

    audio = load_audio(audio_path)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODELS[model_id]["mlx_repo"],
        language=lang,
    )
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language"),
        "segments": [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"].strip(),
            }
            for s in result.get("segments", [])
        ],
    }


# Cache dei modelli faster-whisper caricati in RAM (caricamento costoso: una sola volta)
_fw_models: dict = {}
_fw_lock = threading.Lock()


def _get_fw_model(model_id: str):
    """Carica (o recupera dalla cache) un WhisperModel faster-whisper.

    device="auto" -> CUDA se c'è una GPU NVIDIA, altrimenti CPU.
    compute_type="default" -> int8 su CPU, float16 su GPU.
    """
    with _fw_lock:
        model = _fw_models.get(model_id)
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                MODELS[model_id]["fw_model"], device="auto", compute_type="default"
            )
            _fw_models[model_id] = model
    return model


def _transcribe_faster_whisper(audio_path: str, model_id: str, lang: str | None) -> dict:
    """Backend cross-platform (Windows/Linux/Mac Intel): faster-whisper / CTranslate2."""
    model = _get_fw_model(model_id)
    audio = load_audio(audio_path)
    segments, info = model.transcribe(audio, language=lang)

    # `segments` è un generatore: lo consumiamo una sola volta.
    segments_out = []
    parts = []
    for s in segments:
        parts.append(s.text)
        segments_out.append(
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        )
    return {
        "text": "".join(parts).strip(),
        "language": info.language,
        "segments": segments_out,
    }


# ------------------------------- Parakeet -------------------------------

# Cache dei modelli Parakeet caricati in RAM (caricamento costoso: una sola volta)
_pk_models: dict = {}
_pk_lock = threading.Lock()


def _pyav_load_audio(filename, sampling_rate: int, dtype=None):
    """Sostituto di parakeet_mlx.audio.load_audio: decodifica via PyAV.

    parakeet-mlx decodifica i file chiamando il binario ffmpeg di sistema; qui usiamo
    invece il nostro load_audio (PyAV: librerie ffmpeg dentro la wheel pip), così non
    serve installare nulla a livello di sistema — come per i modelli Whisper.
    """
    import mlx.core as mx

    audio = load_audio(str(filename), sr=sampling_rate)
    array = mx.array(audio)
    return array.astype(dtype) if dtype is not None else array


def _get_parakeet_model(model_id: str):
    with _pk_lock:
        model = _pk_models.get(model_id)
        if model is None:
            import parakeet_mlx.parakeet as _pk_parakeet
            from parakeet_mlx import from_pretrained

            # Patch: niente ffmpeg di sistema, decodifica via PyAV (vedi sopra).
            _pk_parakeet.load_audio = _pyav_load_audio

            model = from_pretrained(MODELS[model_id]["mlx_repo"])
            _pk_models[model_id] = model
    return model


def _transcribe_parakeet(audio_path: str, model_id: str) -> dict:
    """Backend Apple Silicon: NVIDIA Parakeet via parakeet-mlx (GPU Metal).

    Parakeet v3 è multilingue con auto-rilevamento: la lingua non va specificata.

    dtype=float32 obbligatorio: con il default bfloat16 il calcolo del log-mel di
    parakeet-mlx (mx.view complex->dtype) produce feature dimezzate e va in errore
    di shape nel matmul.
    """
    import mlx.core as mx

    model = _get_parakeet_model(model_id)
    result = model.transcribe(audio_path, dtype=mx.float32)
    sentences = getattr(result, "sentences", None) or []
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in sentences
    ]
    text = (getattr(result, "text", "") or "").strip() or " ".join(
        s["text"] for s in segments
    )
    return {
        "text": text,
        "language": getattr(result, "language", None),
        "segments": segments,
    }


# ------------------------------- Export SRT -------------------------------

def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    """Converte i segmenti in formato SRT (sottotitoli)."""
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}\n{seg['text']}\n"
        )
    return "\n".join(blocks)


# ------------------------- Export multi-modello -------------------------

_SEP = "=" * 64
_SEP2 = "#" * 64


def to_multi_txt(ordered_results: list[tuple[str, dict]], filename: str) -> str:
    """Combina le trascrizioni di più modelli in un unico file di testo.

    `ordered_results`: lista di (model_id, result) nell'ordine di esecuzione;
    ogni `result` è il dict di transcribe() con in più la chiave "elapsed",
    oppure {"error": messaggio} se quel modello è fallito.

    Il formato è testo puro con sezioni ben delimitate e senza timestamp:
    pensato per il confronto manuale o per l'analisi automatica con un LLM.
    """
    n = len(ordered_results)
    parts = [
        _SEP,
        f"TRASCRIZIONI MULTIPLE — {filename}",
        f"Modelli eseguiti in sequenza: {n}",
        _SEP,
    ]
    for i, (model_id, res) in enumerate(ordered_results, 1):
        label = MODELS.get(model_id, {}).get("label", model_id)
        error = res.get("error")
        header = f"MODELLO {i}/{n} — {label} [{model_id}]"
        if error:
            header += " — ERRORE"
            body = str(error)
        else:
            meta = []
            if res.get("language"):
                meta.append(f"Lingua: {res['language']}")
            if res.get("elapsed") is not None:
                meta.append(f"Tempo: {res['elapsed']}s")
            if meta:
                header += "\n" + " · ".join(meta)
            body = res.get("text", "")
        parts += ["", _SEP2, header, _SEP2, "", body]
    return "\n".join(parts).rstrip() + "\n"
