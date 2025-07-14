# main.py
import os
import onnxruntime, onnx
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import io
import json
from google.cloud import storage


app = FastAPI()
with open("labels.txt", "r") as file:
    labels = [line.strip() for line in file]


def xywh_to_xyxy(boxes):
    # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def compute_iou(box, boxes):
    # box: (4,), boxes: (N, 4)
    # 경계선 잡기
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter

    return inter / union


def nms_onnx(boxes_xywh, scores, iou_threshold=0.5):
    boxes = xywh_to_xyxy(boxes_xywh)
    indices = scores.argsort()[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        ious = compute_iou(boxes[current], boxes[indices[1:]])
        indices = indices[1:][ious <= iou_threshold]

    return keep


# # ONNX 모델 로딩
# session = onnxruntime.InferenceSession(
#     "gs://k-ai-model-bucket/yolo/yolov8n.onnx", providers=["CPUExecutionProvider"]
# )
def get_session():

    if not os.path.exists("k-ai-model-bucket-yolov8n.onnx"):
        client = storage.Client.from_service_account_json("key.json")
        bucket = client.bucket("k-ai-model-bucket")
        blob = bucket.blob("yolo/yolov8n.onnx")
        print("Downloading ONNX model...")
        blob.download_to_filename("k-ai-model-bucket-yolov8n.onnx")

    return onnxruntime.InferenceSession(
        "k-ai-model-bucket-yolov8n.onnx", providers=["CPUExecutionProvider"]
    )


session = get_session()


def process_outputs(outputs, confidence_threshold=0.5, iou_threshold=0.5):
    result = outputs[0]  # shape: (1, 84, 8400)
    result = result.transpose(0, 2, 1).squeeze(0)  # → (8400, 84)

    boxes = result[:, :4]  # cx, cy, w, h
    class_scores = result[:, 4:]
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    mask = confidences > confidence_threshold

    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    keep = nms_onnx(boxes, confidences, iou_threshold)

    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    results = []
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        results.append(
            {
                "center_x": float(cx),
                "center_y": float(cy),
                "box_width": float(w),
                "box_height": float(h),
                "confidence": float(confidences[i]),
                "label": labels[class_ids[i]],
            }
        )

    return {"length": len(results), "results": results}


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
