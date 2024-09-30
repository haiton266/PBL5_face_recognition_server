import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
import numpy as np

# Set DeepFace environment path
os.environ['DEEPFACE_HOME'] = './model'

def cosine_similarity(vec1, vec2):
    """
    Tính độ tương đồng cosin giữa hai vector.

    Args:
        vec1 (numpy.ndarray): Vector thứ nhất.
        vec2 (numpy.ndarray): Vector thứ hai.

    Returns:
        float: Giá trị độ tương đồng cosin giữa hai vector.
               Trả về giá trị trong khoảng [-1, 1], giá trị càng gần 1 thì hai vector càng tương đồng.
    """
    # Tính tích vô hướng giữa hai vector
    dot_product = np.dot(vec1, vec2)
    
    # Tính độ dài của mỗi vector (norm của vector)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Tránh chia cho 0 bằng cách kiểm tra độ dài vector
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    # Tính toán độ tương đồng cosin
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_sim

def detect_largest_face(image, model):
    # Detect faces in the image:np.array
    results = DeepFace.extract_faces(
        img_path=image, detector_backend=model)
    
    if len(results) == 0:
        return None  # Return None if no faces are found

    # Function to calculate the area of the face based on bounding box
    def face_area(face_info):
        return face_info['facial_area']['w'] * face_info['facial_area']['h']

    # Find the largest face by area
    largest_face = max(results, key=lambda x: face_area(x))

    # Extract the bounding box and confidence of the largest face
    face_info = largest_face['facial_area']

    # Crop the face from the original image
    img = image.copy()
    x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']
    cropped_face_bgr = img[y:y+h, x:x+w]  # Crop the face from the image

    # Convert from BGR to RGB
    cropped_face_rgb = cv2.cvtColor(cropped_face_bgr, cv2.COLOR_BGR2RGB)

    # Return the cropped face in RGB format
    return cropped_face_rgb

def embedding(image_url, model_extract):
    """
    Function to get the embedding of the image
    """
    face_encoding = DeepFace.represent(
        img_path=image_url, model_name=model_extract, detector_backend="skip", align=False, normalization='Facenet')
    face_info = face_encoding[0]['embedding']
    return face_info


# # Call the function
# model_backend = 'yolov8' 
# with open("hai.png", "rb") as image_file:
#     image_data = image_file.read()
# cropped_face = detect_largest_face(image_data, model_backend)


# if cropped_face is not None:
#     # Save the cropped face in RGB format for further use (optional)
#     plt.imsave("cropped_face.jpg", cropped_face)
#     print('Embedded vector: ', embedding("cropped_face.jpg", 'Facenet'))
# else:
#     print("No face detected.")


