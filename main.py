# main.py
import onnxruntime
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import io
import json

app = FastAPI()

# ONNX 모델 로딩
session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])


def process_outputs(outputs, confidence_threshold=0.5):
    # 예시: 바운딩 박스와 클래스 라벨을 처리하는 함수
    # 실제 모델의 출력 형식에 맞게 구현 필요
    result = outputs[0]
    result = result.transpose(0, 2, 1).squeeze(0)
    center_x = result[:, 0]  # 중심 좌표
    center_y = result[:, 1]
    scores = result[:, 4:]
    box_dims = []  # 박스 크기 (너비, 높이)
    box_heights = result[:, 3]
    box_widths = result[:, 2]
    confidences = scores.max(axis=1)
    labels = np.argmax(result, axis=1)  # 가장 높은 확률의 클래스 인덱스
    mask = confidences > confidence_threshold
    centers = centers[mask]  # 중심 좌표 필터링

    results = []
    for i in range(len(result.shape[0])):
        results.append({"center_x": center_x[i], "center_y": int(label)})

    return results


@app.post("/predict")
async def predict(file: UploadFile):
    # 이미지 전처리
    image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((640, 640))
    input_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)

    # 모델 추론
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)

    # 후처리 (예: 바운딩 박스, 클래스 라벨)
    result = process_outputs(outputs)

    return {"predictions": result}
