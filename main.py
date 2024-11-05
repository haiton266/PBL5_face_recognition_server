from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
import requests
import numpy as np
from io import BytesIO
import os
import json
from ultralytics import YOLO

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

# Pydantic model to accept image_url and name
class FaceData(BaseModel):
    image_url: str
    name: str

# Path to save the embeddings (this simulates a database)
DATABASE_PATH = 'face_embeddings.json'

# Helper function to load existing face embeddings from file
def load_face_embeddings():
    if os.path.exists(DATABASE_PATH):
        try:
            with open(DATABASE_PATH, 'r') as file:
                data = json.load(file)
                print(f"Loaded existing data from {DATABASE_PATH}: {data}")
                return data
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            return {"face_data": []}  # Trả về dữ liệu rỗng nếu tệp JSON không hợp lệ
    return {"face_data": []}  # Trả về cấu trúc rỗng nếu tệp không tồn tại

# Helper function to save face embeddings to file
def save_face_embeddings(embeddings):
    try:
        with open(DATABASE_PATH, 'w') as file:
            json.dump(embeddings, file, indent=4)
            print(f"Data successfully saved to {DATABASE_PATH}")
    except Exception as e:
        print(f"Error saving data to JSON file: {e}")


@app.post("/add-face/")
async def process_image_endpoint(data: FaceData):
    """
    Endpoint to process an image from a given URL and save face embedding with the user's name.
    """
    try:
        # Fetch the image from the URL
        print(f"Fetching image from URL: {data.image_url}")
        response = requests.get(data.image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        print("Image fetched successfully")

        # Convert image to RGB if it has an alpha channel (RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = np.array(image)
        
        # Detect the largest face
        try:
            print("Detecting the largest face...")
            largest_face = detect_largest_face(image, 'yolov8')

            if largest_face is None:
                print("No face detected")
                return JSONResponse(status_code=200, content={"result": "No face detected"})

        except Exception as e:
            print("Error during face detection:", str(e))
            return JSONResponse(status_code=200, content={"result": "No face detected", "error": str(e)})

        # Get the embedding of the detected face
        print("Generating embedding for the detected face...")
        face_embedding = embedding(largest_face, 'Facenet')
        print(f"Embedding generated: {face_embedding}")

        # Load existing embeddings from the database (JSON file)
        face_data = load_face_embeddings()
        print(f"Loaded face data: {face_data}")

        # Append the new face embedding
        face_data['face_data'].append({
            "id": data.name,  # Using the name as ID
            "vector": face_embedding  # Embedding vector as list
        })

        # Save the updated embeddings back to the JSON file
        save_face_embeddings(face_data)
        print("Face embedding saved successfully")

        return JSONResponse(content={"result": "Face detected and saved", "name": data.name, "embedding": face_embedding})

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail="Error fetching the image: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the image: " + str(e))



@app.post("/verify-face/")
async def verify_face(image_url: str):
    """
    Endpoint to verify a face from an image URL.
    It compares the face in the image with known embeddings stored in the face_embeddings.json file.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url, allow_redirects=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Convert image to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = np.array(image)

        # Detect the largest face in the image
        try:
            largest_face = detect_largest_face(image, 'yolov8')
            if largest_face is None:
                return JSONResponse(status_code=200, content={"result": "No face detected"})

        except Exception as e:
            print("Error during face detection:", str(e))
            return JSONResponse(status_code=200, content={"result": "No face detected", "error": str(e)})

        # Get the embedding of the detected face
        face_embedding = embedding(largest_face, 'Facenet')

        # Load known face embeddings from the JSON database
        face_data = load_face_embeddings()
        print(f"Loaded existing face data: {face_data}")

        # Ensure the 'face_data' key exists and is a list
        if 'face_data' not in face_data or not isinstance(face_data['face_data'], list):
            raise HTTPException(status_code=500, detail="Invalid face data structure in JSON")

        # Variables to hold the closest match
        closest_id = None
        closest_distance = -1

        # Iterate over each known embedding in face_data and find the closest match
        for face in face_data['face_data']:
            # Check if face contains both 'id' and 'vector'
            if not all(k in face for k in ("id", "vector")):
                print(f"Skipping invalid entry in face_data: {face}")
                continue

            known_id = face["id"]
            known_embedding = np.array(face["vector"])

            # Calculate the distance (e.g., cosine similarity or euclidean distance)
            distance = cosine_similarity(face_embedding, known_embedding)

            # Update the closest match if this one is closer
            if distance > closest_distance and distance > 0.55:  # Adjust threshold as needed
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

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Bạn có thể thay đổi thành mô hình YOLOv8 lớn hơn nếu muốn

class ImageURL(BaseModel):
    image_url: str

@app.post("/detect-human/")
async def detect_human(image_url: str):
    """
    Endpoint để kiểm tra xem có người trong ảnh từ URL của Cloudinary hay không.
    """
    try:
        # Fetch the image from Cloudinary URL
        print(f"Fetching image from Cloudinary URL: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        print("Image fetched successfully")

        # Convert image to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image = np.array(image)

        # Sử dụng YOLO để phát hiện người trong ảnh
        results = model.predict(source=image, classes=0)  # Chỉ kiểm tra lớp "người" (classes=0)

        # Kiểm tra kết quả
        if len(results[0].boxes) > 0:  # Nếu có ít nhất một bounding box cho lớp "người"
            return JSONResponse(content={"result": "Human detected", "num_humans": len(results[0].boxes)})
        else:
            return JSONResponse(content={"result": "Human detected"})

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail="Error fetching the image: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the image: " + str(e))