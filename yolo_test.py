import os
from matplotlib import patches, pyplot as plt
import numpy as np
from PIL import Image
from google.cloud import storage
import onnxruntime
from main import process_outputs


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
# def get_session():

#     if not os.path.exists("k-ai-model-bucket-yolov8n.onnx"):
#         client = storage.Client.from_service_account_json("key.json")
#         bucket = client.bucket("k-ai-model-bucket")
#         blob = bucket.blob("yolo/yolov8n.onnx")
#         print("Downloading ONNX model...")
#         blob.download_to_filename("k-ai-model-bucket-yolov8n.onnx")

#     return onnxruntime.InferenceSession(
#         "k-ai-model-bucket-yolov8n.onnx", providers=["CPUExecutionProvider"]
#     )


session = get_session()


# def process_outputs(outputs, confidence_threshold=0.1, iou_threshold=0.5):
#     result = outputs[0]  # (1, 84, 8400)
#     result = result.transpose(0, 2, 1).squeeze(0)  # → (8400, 84)

#     boxes = result[:, :4]  # cx, cy, w, h
#     class_scores = result[:, 4:]  # no objectness, just class probs

#     confidences = class_scores.max(axis=1)
#     class_ids = class_scores.argmax(axis=1)

#     mask = confidences > confidence_threshold
#     boxes = boxes[mask]
#     confidences = confidences[mask]
#     class_ids = class_ids[mask]

#     keep = nms_onnx(boxes, confidences, iou_threshold)

#     boxes = boxes[keep]
#     confidences = confidences[keep]
#     class_ids = class_ids[keep]

#     results = []
#     for i in range(len(boxes)):
#         cx, cy, w, h = boxes[i]
#         results.append(
#             {
#                 "center_x": float(cx),
#                 "center_y": float(cy),
#                 "box_width": float(w),
#                 "box_height": float(h),
#             }
#         )

#     return {"length": len(results), "results": results}


def show_boxes(file_url: str, confidence_threshold=0.25, iou_threshold=0.5):
    """
    주어진 이미지 파일에서 객체 감지 결과를 시각화하는 함수입니다.

    Args:
        file_url (str): 처리할 이미지 파일의 경로입니다.
        confidence_threshold (float, optional): 감지된 객체의 신뢰도 임계값입니다.
            기본값은 0.25입니다. 이 값보다 낮은 신뢰도를 가진 객체는 무시됩니다.
        iou_threshold (float, optional): Non-Maximum Suppression(NMS)에 사용되는
            IoU(Intersection over Union) 임계값입니다. 기본값은 0.5입니다.

    작업 과정:
        1. 이미지 파일을 읽고 전처리합니다.
        2. 모델을 사용하여 객체 감지를 수행합니다.
        3. 감지된 객체의 바운딩 박스와 클래스 라벨을 후처리합니다.
        4. 결과를 시각화하여 이미지에 바운딩 박스를 그립니다.

    Returns:
        None: 결과는 화면에 시각적으로 표시됩니다.
    """
    # 이미지 전처리
    with open(file_url, "rb") as f:
        image = Image.open(f).convert("RGB").resize((640, 640))

    input_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    # 모델 추론
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    print("output type:", type(outputs[0]))
    # 후처리 (예: 바운딩 박스, 클래스 라벨)
    result = process_outputs(outputs, confidence_threshold, iou_threshold)

    for res in result["results"]:
        print(res)
        rect = patches.Rectangle(
            (
                res["center_x"] - res["box_width"] / 2,
                res["center_y"] - res["box_height"] / 2,
            ),
            res["box_width"],
            res["box_height"],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.axis("off")
    plt.show()


import onnx

model = onnx.load("k-ai-model-bucket-yolov8n.onnx")
print("Inputs:", [input.name for input in model.graph.input])
print("Outputs:", [output.name for output in model.graph.output])

show_boxes(
    "coverim20240404225240_5.jpeg", confidence_threshold=0.05, iou_threshold=0.65
)
# return {"predictions": result}
