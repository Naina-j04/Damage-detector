from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi.responses import JSONResponse, HTMLResponse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/simple_damage_classifier.keras"
CLASS_INDEX_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


app = FastAPI(
    title="Product Damage Detector API",
    description="API for detecting damage in product images using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    logger.info("Loading model and class indices...")
    model = load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    logger.info("Model and class indices loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or class indices: {str(e)}")
    raise

def validate_image(file: UploadFile) -> None:
    """Validate the uploaded image file."""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
        
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  
    size = file.file.tell()
    file.file.seek(0)  
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
        )

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that redirects to documentation."""
    return """
    <html>
        <head>
            <title>Product Damage Detector API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 { color: #2c3e50; }
                .links {
                    margin-top: 20px;
                }
                .links a {
                    display: inline-block;
                    margin-right: 20px;
                    color: #3498db;
                    text-decoration: none;
                }
                .links a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to Product Damage Detector API</h1>
            <p>This API provides endpoints for detecting damage in product images using deep learning.</p>
            <div class="links">
                <a href="/docs">API Documentation (Swagger UI)</a>
                <a href="/redoc">API Documentation (ReDoc)</a>
            </div>
        </body>
    </html>
    """

# ----- Prediction Endpoint -----
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    start_time = datetime.now()
    temp_path = None
    
    try:
        # Validate input
        validate_image(file)
        
        # Create temporary file
        temp_path = f"temp_{file.filename}"
        logger.info(f"Processing image: {file.filename}")

        # Save uploaded image temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess image
        img = image.load_img(temp_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        predicted_label = 1 if prediction >= 0.5 else 0
        class_name = index_to_class[predicted_label]
        confidence = prediction if predicted_label == 1 else 1 - prediction

        # Map to result label
        if confidence < 0.65:
            result = "uncertain"
        elif class_name == "damaged":
            result = "damaged"
        else:
            result = "not damaged"

        # Log prediction
        logger.info(f"Prediction for {file.filename}: {result} (confidence: {confidence:.2f})")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "prediction": result,
                "confidence": f"{confidence * 100:.2f}%",
                "processing_time": f"{(datetime.now() - start_time).total_seconds():.2f}s"
            }
        )
    except HTTPException as he:
        logger.warning(f"Validation error: {str(he)}")
        return JSONResponse(
            status_code=he.status_code,
            content={
                "status": "error",
                "detail": str(he.detail)
            }
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e)
            }
        )
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"Cleaned up temporary file: {temp_path}")

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }
    )
