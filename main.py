# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from yolo.yolo_fun import yolo_detect_and_convert_bbox
from clip.clip_fun import get_clip_embedding_from_pil, get_clip_session
from sklearn.preprocessing import normalize
import numpy as np

app = FastAPI()
clip_session = get_clip_session()


@app.get("/health")
def health():
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}


@app.post("/process")
async def process_image(file: UploadFile = File(None), url: str = Form(None)):
    """
    이미지 파일 업로드 또는 URL로부터 이미지를 받아 YOLO+CLIP 추론 결과를 반환합니다.
    - file: 업로드 이미지
    - url: 이미지 URL
    """
    try:
        # 1. 이미지 로딩 (업로드 파일 또는 URL)
        image = await load_image(file, url)
        if image is None:
            return JSONResponse(
                status_code=400, content={"detail": "No image file was uploaded"}
            )

        # 2. YOLO 추론 및 bbox 변환 (원본 크기 기준)
        yolo_results = yolo_detect_and_convert_bbox(image)

        # 3. 객체별 crop 및 CLIP 임베딩 추출
        results = []
        for obj in yolo_results:
            crop_img = image.crop(obj["crop_box"])
            clip_emb = get_clip_embedding_from_pil(crop_img, clip_session)
            clip_emb = np.array(clip_emb).flatten().reshape(1, -1)
            clip_emb = normalize(clip_emb)[0].tolist()
            results.append(
                {
                    "label": obj["label"],
                    "confidence": obj["confidence"],
                    "bbox": obj["bbox"],
                    "clip_embedding": clip_emb,
                }
            )
        print(len(clip_emb), "차원 CLIP 임베딩 벡터 생성 완료")
        return JSONResponse(content=results)
    except Exception:
        import traceback

        print(traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"detail": "Failed to process image"}
        )


async def load_image(file, url):
    """
    업로드 파일 또는 URL로부터 이미지를 로드하여 PIL Image로 반환합니다.
    실패 시 None 반환.
    """
    if file:
        file_bytes = await file.read()
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    elif url:
        import requests

        response = requests.get(url)
        if response.status_code != 200:
            return None
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        return None
