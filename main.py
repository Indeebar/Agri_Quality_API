from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

# Allow all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
from keras.models import load_model

model = load_model("saved_model")


# Manually define class labels in order (use your real order here)
class_labels = [
    "Banana_Stems_Contaminated",
    "Banana_Stems_Dry",
    "Banana_Stems_Moisturized",
    "Coconut_Shells_Dry",
    "Maize_Stalks_Dry",
    "Rice_Straw_Dry",
    "Sugarcane_Bagasse_Dry"
]  # Modify this list to match your training generator order

@app.get("/")
def home():
    return {"message": "Agri Waste Quality Classifier API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))  # Resize to match model input

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize like training

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_index] * 100
    predicted_label = class_labels[predicted_class_index]

    # Return JSON response
    return {
        "predicted_class": predicted_label,
        "confidence_percent": f"{confidence:.2f}%",
        "numeric_confidence": round(float(confidence), 2)
    }
