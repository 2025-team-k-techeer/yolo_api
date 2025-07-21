import onnxruntime
from PIL import Image
from clip.clip_fun import (
    get_clip_session,
    preprocess_image,
)  # 위 함수들이 있는 파일명에 맞게 수정

# 1. 세션 로드
session = get_clip_session()

# 2. 테스트 이미지 로딩 (로컬 이미지 파일)
img = Image.open("test_image.jpg").convert("RGB")
input_tensor = preprocess_image(img)

# 3. 입력 이름 확인
input_name = session.get_inputs()[0].name
print(f"입력 이름: {input_name}")

# 4. 추론
outputs = session.run(None, {input_name: input_tensor})
print("임베딩 벡터 shape:", outputs[0].shape)
print("일부 값:", outputs[0][0][:5])
