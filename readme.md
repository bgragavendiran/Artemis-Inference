# ARTEMIS Inference
This API, built with FastAPI, allows users to perform inference on images using a trained model for flower count detection. The API provides two endpoints for image inference, either by uploading an image or referencing an image URL stored in Firebase.

## Prerequisites
- Firebase project setup with Realtime Database and Authentication.
- Firebase Admin credentials file (artemis-f901d-firebase-adminsdk.json).
- Docker for containerized deployment (optional).

## Endpoints
1. /infer-from-firebase
   This endpoint fetches images for a specified plant from Firebase, performs inference, and updates the count back to Firebase.

### Method: POST
#### Body Parameters:

> plant_name: Name of the plant.
firebase_uid: Firebase user ID for authentication.
secret_key: Shared secret for additional authentication.
>

Example Curl Request:
```
curl -X POST "http://localhost:8080/infer-from-firebase" \
-F "plant_name=Bean Plant" \
-F "firebase_uid=YOUR_FIREBASE_UID" \
-F "secret_key=88a1fc2d2ac751709328b2f10d43d9f270eb08becb55a35362e93806f953ce0f"
2. /infer-from-upload
   This endpoint allows users to upload an image directly for inference. The result will return the flower count in the response.
```

### Method: POST
#### Body Parameters:

>file: Image file to be uploaded.
firebase_uid: Firebase user ID for authentication.
secret_key: Shared secret for additional authentication.

Example Curl Request:

```
curl -X POST "http://localhost:8080/infer-from-upload" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/image.jpg" \
-F "firebase_uid=YOUR_FIREBASE_UID" \
-F "secret_key=88a1fc2d2ac751709328b2f10d43d9f270eb08becb55a35362e93806f953ce0f"
```

# Authentication
Firebase UID: The firebase_uid parameter ensures that only authenticated users can access the API.
Secret Key: An additional layer of security using a shared secret key.

## Running the Service

### Local Setup:
Install dependencies and run the FastAPI app locally.
- Docker:
- - Build and run the Docker container.
- - Ensure port 8080 is used for compatibility with Google Cloud Run.
## Cloud Run :
Image is also pushed to GCP Container Registry
Final Endpoints in Cloud Run are
https://flower-count-api-79045011025.us-central1.run.app

## Improvements to be Considered:
- In future will have to implement a OAUTH 2.0 or SAML on top of this app to improve security on the endpoint
- The current Endpoint works on a Symmetric Key. This will be migrated to GOOGLE SECRET MANAGER or AWS SECRET MANAGER to abstract the keys.
- The Docker registry now holds a raw copy of private key, but the image is airgapped so no worries as of now but would have to move that to SECRET MANAGER as well.