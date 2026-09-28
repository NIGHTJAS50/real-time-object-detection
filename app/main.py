from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import tempfile

app = FastAPI(title="Real-Time Object Detection")

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    data = await image.read()
    with tempfile.NamedTemporaryFile(suffix=Path(image.filename or ".jpg").suffix, delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        from ultralytics import YOLO
        results = YOLO("yolov8n.pt")(path, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({"label": result.names[int(box.cls)], "confidence": round(float(box.conf), 4), "box": [round(float(x), 2) for x in box.xyxy[0]]})
        return {"detections": detections}
    finally:
        Path(path).unlink(missing_ok=True)
