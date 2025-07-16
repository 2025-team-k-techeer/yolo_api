# main.py
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import io
from yolo.yolo_fun import get_yolo_session, process_outputs

from clip.clip_fun import get_clip_embedding_from_uploadfile, get_clip_session

app = FastAPI()


yollo_session = get_yolo_session()
clip_session = get_clip_session()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/yolo/predict")
async def predict(file: UploadFile):
    # 이미지 전처리
    image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((640, 640))
    input_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)

    # 모델 추론
    inputs = {yollo_session.get_inputs()[0].name: input_tensor}
    outputs = yollo_session.run(None, inputs)

    # 후처리 (예: 바운딩 박스, 클래스 라벨)
    result = process_outputs(outputs)

    return {"predictions": result}


@app.post("/clip/embedding")
async def clip_embed(image: UploadFile):
    embedding = get_clip_embedding_from_uploadfile(image, clip_session)
    vector = embedding[0]  # 첫 번째 벡터 (batch에서 꺼내기)
    return {
        "embedding": vector.tolist(),  # 리스트 형태로 반환
        "length": len(vector),  # 벡터 길이 (예: 512)
    }


@app.on_event("startup")
def print_clip_model_info():
    print("CLIP 입력 이름:", clip_session.get_inputs()[0].name)
    print("CLIP 출력 이름:", clip_session.get_outputs()[0].name)
