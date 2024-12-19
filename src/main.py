from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
import base64
from PIL import Image
from deepface import DeepFace
import oci
import numpy as np

app = FastAPI()

# Oracle Cloud Storage Config
NAMESPACE = "grxsx54cyxpu"
BUCKET_NAME = "faces-bucket"
CONFIG_PATH = "/app/.oci/config"
PROFILE_NAME = "DEFAULT"


def initialize_client(config_file_path=CONFIG_PATH, profile_name=PROFILE_NAME):
    """Initialize the OCI client configuration."""
    config = oci.config.from_file(
        file_location=config_file_path, profile_name=profile_name
    )
    return config


def upload_image_to_oci(config, namespace, bucket_name, object_name, image_base64):
    """Upload an image to Oracle Cloud Storage."""
    object_storage_client = oci.object_storage.ObjectStorageClient(config)
    image_bytes = base64.b64decode(image_base64)
    object_storage_client.put_object(namespace, bucket_name, object_name, image_bytes)


def download_image_from_oci(config, namespace, bucket_name, object_name):
    """Download an image from Oracle Cloud Storage."""
    object_storage_client = oci.object_storage.ObjectStorageClient(config)
    response = object_storage_client.get_object(namespace, bucket_name, object_name)
    image_bytes = response.data.content
    return base64.b64encode(image_bytes).decode("utf-8")


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


@app.post("/cadastrar/")
def save_face(request: SaveFaceRequest):
    try:
        config = initialize_client()
        object_name = f"{request.name}.jpg"

        # upload imagem para bucket
        upload_image_to_oci(
            config, NAMESPACE, BUCKET_NAME, object_name, request.image_base64
        )

        return {"message": f"Rosto salvo com sucesso para {request.name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro salvando rosto: {str(e)}")


@app.post("/identificar/")
def recognize_face(request: RecognizeFaceRequest):
    try:
        config = initialize_client()

        # decode imagem base64
        query_image = decode_base64_image(request.image_base64)

        # salvar imagem do rosto que quer identificar em memoria
        buffer = BytesIO()
        query_image.save(buffer, format="JPEG")
        buffer.seek(0)
        query_image_np = np.array(Image.open(buffer))

        # baixar imagens do bucket para comparacao
        object_storage_client = oci.object_storage.ObjectStorageClient(config)
        objects = object_storage_client.list_objects(
            NAMESPACE, BUCKET_NAME
        ).data.objects

        for obj in objects:
            object_name = obj.name
            stored_image_base64 = download_image_from_oci(
                config, NAMESPACE, BUCKET_NAME, object_name
            )
            stored_image = decode_base64_image(stored_image_base64)
            stored_image_np = np.array(stored_image)

            # reconhecimento facial usando deepface com as imagens do bucket
            result = DeepFace.verify(
                query_image_np, stored_image_np, enforce_detection=False
            )

            if result["verified"]:
                name = object_name.split(".")[0]
                return {"message": "Rosto reconhecido", "nome": name}

        return {"message": "Nenhuma pessoa encontrada"}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erro reconhecendo rosto: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
