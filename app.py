import io, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

BASE = Path(__file__).parent
MAX_BYTES = 8 * 1024 * 1024          # batas unggahan 8 MB
NEGATIVE = "Negatif"

model = tf.keras.models.load_model(BASE / "model" / "scantik_model_v2.keras")
with open(BASE / "model" / "class_names_v2.json", encoding="utf-8") as f:
    class_names = json.load(f)

motif_path = BASE / "motif_data.json"      # diisi di langkah berikutnya
motif_data = {}
if motif_path.exists():
    with open(motif_path, encoding="utf-8") as f:
        motif_data = json.load(f)

app = FastAPI(title="ScanTik Pekalongan API")


def error(message: str, code: int):
    return JSONResponse(status_code=code,
                        content={"status": "error", "message": message, "code": code})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw or len(raw) > MAX_BYTES:
        return error("Ukuran file tidak valid (maks 8 MB).", 400)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError):
        return error("Gambar tidak valid atau format file tidak didukung.", 400)

    # Preprocessing MobileNetV2 sudah ada di dalam model: kirim piksel mentah 0-255
    x = np.asarray(img, dtype=np.float32)[None, ...]
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    label, p = class_names[i], float(probs[i])

    if label == NEGATIVE:
        details = {"name": "Motif tidak dikenali", "origin": None,
                   "philosophy": None, "characteristics": None}
    else:
        details = motif_data.get(label, {})

    return {
        "status": "success",
        "data": {
            "predicted_class": label,
            "confidence_score": round(p * 100, 2),
            "confidence_decimal": round(p, 4),
            "motif_details": details,
        },
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }