from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
import requests
import numpy as np
from io import BytesIO
import os

from utils import detect_largest_face, embedding, cosine_similarity
app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

@app.post("/add-face/")
async def process_image_endpoint(image_url: str):
    """
    Endpoint to process an image from a given URL.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Convert image to RGB if it has an alpha channel (RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = np.array(image)
        
        largest_face = detect_largest_face(image, 'yolov8')

        if largest_face is None:
            return JSONResponse(content={"result": "No face detected"})

        # Get the embedding of the detected face
        face_embedding = embedding(largest_face, 'Facenet')

        return JSONResponse(content={"result": "Face detected", "embedding": face_embedding})

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail="Error fetching the image: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the image: " + str(e))

@app.post("/verify-face/")
async def verify_face(image_url: str, face_data: dict):
    """
    Endpoint to verify a face from an image URL.
    The request body includes a JSON object with {id, vector} for each known embedding.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Convert image to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = np.array(image)

        # Detect the largest face
        largest_face = detect_largest_face(image, 'yolov8')

        if largest_face is None:
            return JSONResponse(content={"result": "No face detected"})

        # Get the embedding of the detected face
        face_embedding = embedding(largest_face, 'Facenet')

        # List to hold the distances between the current face and known faces
        closest_id = None
        closest_distance = -1

        # Iterate over each known embedding in face_data and find the closest match
        for face in face_data['face_data']:
            known_id = face["id"]
            known_embedding = np.array(face["vector"])

            # Calculate the distance (you can use cosine similarity or euclidean distance)
            distance = cosine_similarity(face_embedding, known_embedding)

            # Update the closest match if this one is closer
            if distance > closest_distance and distance > 0.25:
                closest_distance = distance
                closest_id = known_id

        # Return the closest match
        if closest_id is not None:
            return JSONResponse(content={"result": "Face matched", "closest_id": closest_id, "distance": closest_distance})
        else:
            return JSONResponse(content={"result": "No matching face found"})

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail="Error fetching the image: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the image: " + str(e))