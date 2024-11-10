from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from transformers import DetrForObjectDetection, DetrImageProcessor
from firebase_admin import auth, db, initialize_app, credentials
import requests
import torch
from PIL import Image
from io import BytesIO

# Initialize Firebase Admin
cred = credentials.Certificate("artemis-f901d-firebase-adminsdk-nxntk-638cd0d526.json")
initialize_app(cred, {"databaseURL": "https://artemis-f901d-default-rtdb.asia-southeast1.firebasedatabase.app"})

# Initialize FastAPI and Model
app = FastAPI()
model = DetrForObjectDetection.from_pretrained("smutuvi/flower_count_model")
processor = DetrImageProcessor.from_pretrained("smutuvi/flower_count_model")

# Shared secret key for API authentication
SHARED_SECRET_KEY = "88a1fc2d2ac751709328b2f10d43d9f270eb08becb55a35362e93806f953ce0f"

class InferenceRequest(BaseModel):
    image_url: str
    firebase_uid: str
    secret_key: str

def authenticate_request(firebase_uid: str, secret_key: str):
    if secret_key != SHARED_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret key.")
    try:
        auth.get_user(firebase_uid)  # Verifies the Firebase UID
    except:
        raise HTTPException(status_code=403, detail="Invalid Firebase UID.")


def infer_flower_count(image: Image, threshold=0.5):
    # Process the image through the model
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Apply softmax to get the confidence scores for each query's predictions
    probs = outputs.logits.softmax(-1)
    # Filter detections based on threshold and exclude background class '91'
    relevant_detections = (probs[..., :-1].max(-1).values > threshold) & (probs.argmax(-1) != 91)

    # Count the number of true detections
    flower_count = relevant_detections.sum().item()

    return flower_count

@app.post("/infer-from-firebase")
async def infer_from_firebase(plant_name: str = Form(...), firebase_uid: str = Form(...), secret_key: str = Form(...)):
    authenticate_request(firebase_uid, secret_key)

    # Access Firebase RTDB to retrieve image URLs for the given plant name
    ref_path = f"images/{plant_name}"
    plant_data = db.reference(ref_path).get()

    if not plant_data:
        raise HTTPException(status_code=404, detail="No images found for this plant.")

    # Iterate over each image entry, download the image, and perform inference
    results = {}
    for date_key, date_data in plant_data.items():
        for folder_key, folder_data in date_data.items():
            for image_key, image_info in folder_data.items():
                if image_info.get("status") == "uploaded":
                    image_url = image_info.get("url")
                    try:
                        # Download image
                        response = requests.get(image_url)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert("RGB")

                        # Perform inference
                        flower_count = infer_flower_count(image)

                        # Save result back to Firebase RTDB
                        db.reference(f"{ref_path}/{date_key}/{folder_key}/{image_key}").update({
                            "inference_count": flower_count,
                            "inference_status": "completed"
                        })

                        # Append result for response
                        results[image_key] = {"flower_count": flower_count}
                    except Exception as e:
                        db.reference(f"{ref_path}/{date_key}/{folder_key}/{image_key}").update({
                            "inference_status": "error",
                            "error_message": str(e)
                        })

    return {"status": "inference completed", "results": results}

@app.post("/infer-from-upload")
async def infer_from_upload(file: UploadFile = File(...), firebase_uid: str = Form(...), secret_key: str = Form(...)):
    authenticate_request(firebase_uid, secret_key)

    # Read and prepare image
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    # Perform inference
    flower_count = infer_flower_count(image)
    return {"flower_count": flower_count}

