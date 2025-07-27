import numpy as np
import onnxruntime
import os
from google.cloud import storage
from PIL import Image

# 라벨 파일 로드
# with open("labels.txt", "r") as file:
#     labels = [line.strip() for line in file]

class NoResultsFoundException(Exception):
    """Raised when an expected search yields no results."""
    pass


def get_yolo_session():
    """
    YOLO ONNX 모델 세션을 반환합니다. 없으면 GCS에서 다운로드합니다.
    """
    if not os.path.exists("yolo/yolov8x-worldv2.onnx"):
        client = storage.Client()
        bucket = client.bucket("k-ai-model-bucket")
        blob = bucket.blob("yolo/yolov8x-worldv2.onnx")
        print("Downloading YOLO model...")
        blob.download_to_filename("yolo/yolov8x-worldv2.onnx")
        print("Download Complete ✅")
    return onnxruntime.InferenceSession(
        "yolo/yolov8x-worldv2.onnx", providers=["CPUExecutionProvider"]
    )
def read_class_embeddings(embed_path):
    if not os.path.exists(embed_path):
        client = storage.Client()
        bucket = client.bucket("k-ai-model-bucket")
        blob = bucket.blob(embed_path)
        print("Downloading class embeddings...")
        blob.download_to_filename(embed_path)
        print("Download Complete ✅")
    data = np.load(embed_path)
    return data["class_embeddings"], data["class_list"].tolist()
def inference(inputs):
        # start = time.perf_counter()
        
        outputs = session.run(output_names = [output.name for output in session.get_outputs()], input_feed=
                                   inputs)

        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs
def process_outputs(outputs, confidence_threshold=0.3, iou_threshold=0.5):
    """
    YOLO 모델의 원시 출력값을 후처리하여 객체별 정보 리스트로 반환합니다.
    - confidence_threshold: 신뢰도 임계값
    - iou_threshold: NMS IoU 임계값
    """
    result = outputs[0]
    result = result.transpose(0, 2, 1).squeeze(0)
    boxes = result[:, :4]  # (cx, cy, w, h)
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
    # allowed_labels = set(labels)
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        label = labels[class_ids[i]]
        # if label not in allowed_labels:
        #     continue
        results.append(
            {
                "center_x": float(cx),
                "center_y": float(cy),
                "box_width": float(w),
                "box_height": float(h),
                "confidence": float(confidences[i]),
                "label": label,
            }
        )
    # if YOLO fails to find the boxes, raise the exception. 
    if not results:
        raise NoResultsFoundException("Unable to detect furniture objects.")
    return results


def xywh_to_xyxy(boxes):
    """
    (cx, cy, w, h) → (x1, y1, x2, y2) 변환
    """
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def compute_iou(box, boxes):
    """
    단일 box와 여러 box의 IoU 계산
    """
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
    """
    NMS(Non-Maximum Suppression) 적용
    """
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
session = get_yolo_session()
class_embeddings, labels = read_class_embeddings('yolo/class_embeddings.npz')

def yolo_detect_and_convert_bbox(image: Image.Image):
    """
    입력 이미지를 YOLO로 추론하고, bbox를 원본 크기 기준 [x, y, width, height]로 변환하여 반환합니다.
    crop_box는 crop용 좌표 (int, 이미지 경계 보정)
    """

    orig_w, orig_h = image.width, image.height
    yolo_img = image.resize((640, 640))
    input_array = np.array(yolo_img).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)
    inputs = {session.get_inputs()[0].name: input_tensor, session.get_inputs()[1].name: class_embeddings}
    outputs = inference(inputs)
    objects = process_outputs(outputs)
    results = []
    for obj in objects:
        cx, cy, w, h = (
            obj["center_x"],
            obj["center_y"],
            obj["box_width"],
            obj["box_height"],
        )
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        abs_cx = cx * scale_x
        abs_cy = cy * scale_y
        abs_w = w * scale_x
        abs_h = h * scale_y
        abs_x = abs_cx - abs_w / 2
        abs_y = abs_cy - abs_h / 2
        bbox = [abs_x, abs_y, abs_w, abs_h]  # [x, y, width, height]
        crop_x1 = max(0, int(round(abs_x)))
        crop_y1 = max(0, int(round(abs_y)))
        crop_x2 = min(orig_w, int(round(abs_x + abs_w)))
        crop_y2 = min(orig_h, int(round(abs_y + abs_h)))
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        results.append(
            {
                "label": obj["label"],
                "confidence": obj["confidence"],
                "bbox": bbox,
                "crop_box": crop_box,
            }
        )
    return results

