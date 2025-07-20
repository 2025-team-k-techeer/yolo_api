# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from typing import Optional
import numpy as np
import io
import requests
from yolo.yolo_fun import get_yolo_session, process_outputs
from clip.clip_fun import (
    get_clip_embedding_from_uploadfile,
    get_clip_embedding_from_pil,
    get_clip_session,
)

app = FastAPI()


yollo_session = get_yolo_session()
clip_session = get_clip_session()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/yolo/predict")
async def predict(
    file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)
):
    # Load image from UploadFile
    if file:
        image = (
            Image.open(io.BytesIO(await file.read())).convert("RGB").resize((640, 640))
        )
    # Or load image from URL
    elif url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = (
                Image.open(io.BytesIO(response.content))
                .convert("RGB")
                .resize((640, 640))
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch image from URL: {e}"
            )
    else:
        raise HTTPException(status_code=422, detail="Provide either a file or a URL")

    # Preprocess
    input_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)

    # Model inference
    inputs = {yollo_session.get_inputs()[0].name: input_tensor}
    outputs = yollo_session.run(None, inputs)

    # Postprocessing
    result = process_outputs(outputs)

    return {"predictions": result}


# @app.post("/yolo/predict")
# async def predict(file_link: str):
#     # 이미지 전처리
#     image = Image.open(file_link).convert("RGB").resize((640, 640))
#     input_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
#     input_tensor = np.expand_dims(input_array, axis=0)

#     # 모델 추론
#     inputs = {yollo_session.get_inputs()[0].name: input_tensor}
#     outputs = yollo_session.run(None, inputs)

#     # 후처리 (예: 바운딩 박스, 클래스 라벨)
#     result = process_outputs(outputs)

#     return {"predictions": result}


@app.post("/clip/embedding")
async def clip_embed(
    image: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)
):
    if image:
        embedding = await get_clip_embedding_from_uploadfile(image, clip_session)

    elif url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            embedding = get_clip_embedding_from_pil(img, clip_session)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to load image from URL: {e}"
            )

    else:
        raise HTTPException(status_code=422, detail="Provide either a file or a URL")

    return {"embedding": embedding.tolist(), "length": len(embedding)}
