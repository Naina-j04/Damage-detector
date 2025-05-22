
## Model Architecture

- **Base**: MobileNetV2 (pre-trained, `include_top=False`)
- **Custom Head**: GlobalAveragePooling2D → Dense(512, relu) → BatchNorm → Dropout(0.5) → Dense(256, relu) → BatchNorm → Dropout(0.3) → Dense(1, sigmoid)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy, AUC, Precision, Recall

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd s24_dataset
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the model from scratch (ensure your dataset is in `simple_dataset/`):

```bash
python Training_model.py
```

The trained model and class indices will be saved in the `models/` directory.

## Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

- Access the interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs)
- Use the `/predict/` endpoint to upload an image and get a damage prediction.

## Making Predictions (Standalone Script)

You can also use `prediction_model.py` to predict a single image:

```python
from prediction_model import predict_image

result, confidence = predict_image("path/to/image.jpg")
print(f"Prediction: {result}, Confidence: {confidence:.2f}")
```

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies:
  - tensorflow==2.15.0
  - fastapi, uvicorn, numpy, pillow, python-multipart

## Output

- Trained model: `models/simple_damage_classifier.keras`
- Class mapping: `models/class_indices.json`
- Logs: `app.log`
