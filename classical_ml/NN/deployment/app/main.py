from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from io import BytesIO

from app.inference import ImageClassifier


app = FastAPI(
    title="Intel Scene Classification API",
    version="1.0.0",
)


classifier = ImageClassifier(
    "/app/models/colorjitter_cnn.pth"
)

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 1. Validate file type
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    # 2. Read file
    contents = await file.read()

    # 3. Validate image
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    # 4. Run inference
    try:
        result = classifier.predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}"
        )

    # 5. Return prediction
    return result