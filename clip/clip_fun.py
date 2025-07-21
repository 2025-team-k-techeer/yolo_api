import io
from fastapi import UploadFile
import onnxruntime
import numpy as np
from PIL import Image
from google.cloud import storage
import os
from io import BytesIO


# CLIP 모델을 불러오는 함수
def get_clip_session():
    """
    CLIP ONNX 모델 세션을 반환합니다. 없으면 GCS에서 다운로드합니다.
    """
    if not os.path.exists("clip/image_clip.onnx"):
        client = storage.Client()
        bucket = client.bucket("k-ai-model-bucket")
        blob = bucket.blob("clip/image_clip.onnx")
        print("Downloading CLIP model...")
        blob.download_to_filename("clip/image_clip.onnx")
    return onnxruntime.InferenceSession(
        "clip/image_clip.onnx", providers=["CPUExecutionProvider"]
    )


async def get_clip_embedding_from_uploadfile(
    upload_file: UploadFile, session
) -> np.ndarray:
    image = Image.open(BytesIO(await upload_file.read())).convert("RGB")
    return get_clip_embedding_from_pil(image, session)


def get_clip_embedding_from_pil(image: Image.Image, session) -> np.ndarray:
    """
    crop된 PIL 이미지를 CLIP 모델에 입력하여 임베딩 벡터(768차원)를 반환합니다.
    """
    image_arr = preprocess_image(image)
    image_arr = np.expand_dims(image_arr, axis=0)
    inputs = {"pixel_values": image_arr}
    outputs = session.run(None, inputs)
    return outputs[1]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    CLIP 입력에 맞게 이미지를 224x224로 리사이즈 및 정규화합니다.
    """
    img = image.convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np
