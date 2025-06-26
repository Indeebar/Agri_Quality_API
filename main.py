from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Enable CORS (for development, allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load your .h5 model (must be in project root)
model = load_model("agri_waste_quality_classifier.h5", compile=False)

# ✅ Define your class labels (based on your training)
class_labels = [
    "Banana_Stems_Contaminated",
    "Banana_Stems_Dry",
    "Banana_Stems_Moisturized",
    "Coconut_Shells_Dry",
    "Maize_Stalks_Dry",
    "Rice_Straw_Dry",
    "Sugarcane_Bagasse_Dry"
]

@app.get("/")
def home():
    return {"message": "Agri Waste Quality Classifier API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_index] * 100
        predicted_label = class_labels[predicted_index]

        return {
            "predicted_class": predicted_label,
            "confidence_percent": f"{confidence:.2f}%",
            "numeric_confidence": round(float(confidence), 2)
        }

    except Exception as e:
        return {"error": str(e)}
