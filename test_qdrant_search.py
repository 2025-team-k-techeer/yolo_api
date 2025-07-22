from PIL import Image
from clip.clip_fun import get_clip_session, get_clip_embedding_from_pil
from qdrant_client import QdrantClient, models
import numpy as np

# 1. CLIP ONNX 세션 준비
session = get_clip_session()
print("✅ ONNX 출력 이름:", [output.name for output in session.get_outputs()])

# 2. 이미지 로드
image_path = "image/chair.jpg"
image = Image.open(image_path).convert("RGB")

# 3. CLIP 임베딩 추출 및 L2 정규화
embedding = get_clip_embedding_from_pil(image, session)


# 4. Qdrant 검색 실행
client = QdrantClient(host="localhost", port=6333)

try:
    results = client.search(
        collection_name="furniture_embeddings",  # ✅ 실제 사용 컬렉션명
        query_vector=embedding.tolist(),
        limit=5,
        with_payload=True,
    )

    # 5. 결과 출력
    print("\n🔍 Qdrant 유사 이미지 검색 결과:")
    for rank, point in enumerate(results, 1):
        image_url = point.payload.get("image_url", "N/A")
        print(f"{rank}. 점수: {point.score:.4f}, 이미지 URL: {image_url}")

except Exception as e:
    print(f"[ERROR] Qdrant 검색 실패: {e}")
