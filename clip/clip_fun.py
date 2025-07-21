import io
import onnxruntime
import numpy as np
from PIL import Image
from google.cloud import storage
import os
from fastapi import UploadFile

# from torchvision import transforms
from io import BytesIO


# CLIP 모델을 불러오는 함수
def get_clip_session():
    """
    get_clip_session 함수는 CLIP 모델을 로드하거나 다운로드하여 ONNX Runtime 세션을 반환합니다.

    이 함수는 다음 작업을 수행합니다:
    1. 로컬 디렉토리에 "image_clip.onnx" 파일이 존재하지 않는 경우:
            - Google Cloud Storage에서 "clip.onnx" 파일을 다운로드합니다.
            - 다운로드를 위해 "k-ai-model-bucket" 버킷과 "clip.onnx" 블롭(blob)을 사용합니다.
    2. ONNX Runtime을 사용하여 "image_clip.onnx" 파일로부터 추론 세션을 생성합니다.

    반환값:
             onnxruntime.InferenceSession: CLIP 모델의 추론 세션 객체.

    주의사항:
    - 이 함수는 CPU에서 실행되도록 설정되어 있으며, "CPUExecutionProvider"를 사용합니다.
    - Google Cloud Storage에 접근하려면 적절한 인증이 필요합니다.
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
    image_arr = preprocess_image(image)
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension
    inputs = {"pixel_values": image_arr}
    outputs = session.run(None, inputs)
    return outputs[0]


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np
