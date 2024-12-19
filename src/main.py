from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
import base64
from PIL import Image
from deepface import DeepFace
import os
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Directory to store saved face images
FACE_DATA_DIR = "faces/"
os.makedirs(FACE_DATA_DIR, exist_ok=True)


# Route Models
class SaveFaceRequest(BaseModel):
    name: str
    image_base64: str


class RecognizeFaceRequest(BaseModel):
    image_base64: str


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a base64 image string into a Pillow Image."""
    try:
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


# Route for saving a new face
@app.post("/cadastrar/")
def save_face(request: SaveFaceRequest):
    try:
        # Decode base64 image and save it
        image = decode_base64_image(request.image_base64)
        image_path = os.path.join(FACE_DATA_DIR, f"{request.name}.jpg")
        image.save(image_path)

        return {"message": f"Face saved successfully for {request.name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving face: {str(e)}")


# Route for recognizing a face
@app.post("/identificar/")
def recognize_face(request: RecognizeFaceRequest):
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)

        # Save image to memory (BytesIO)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        image_np = np.array(Image.open(buffer))

        # Perform recognition using DeepFace with in-memory image
        result = DeepFace.find(img_path=image_np, db_path=FACE_DATA_DIR)

        if len(result) > 0:
            recognized_face = result[0].iloc[0]["identity"]
            name = os.path.splitext(os.path.basename(recognized_face))[0]
            return {"message": "Face recognized", "name": name}
        else:
            return {"message": "No match found"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error recognizing face: {str(e)}")


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
