# import onnxruntime
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from io import BytesIO


# # CLIP 모델을 불러오는 함수
# def get_clip_session():
#     return onnxruntime.InferenceSession(
#         "models/clip.onnx", providers=["CPUExecutionProvider"]
#     )


# # 이미지 전처리 함수 (ViT-B/32 기준)
# def preprocess_image(image: Image.Image) -> np.ndarray:
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize(224, interpolation=Image.BICUBIC),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),  # [0, 1]
#             transforms.Normalize(
#                 mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]
#             ),
#         ]
#     )
#     image_tensor = preprocess(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
#     return image_tensor.numpy()


# def get_clip_embedding_from_uploadfile(upload_file, session):
#     # 1. 업로드된 파일 내용을 메모리로 읽음
#     image_bytes = BytesIO(upload_file.file.read())

#     # 2. PIL로 이미지 열기
#     img = Image.open(image_bytes).convert("RGB")
#     img = img.resize((640, 640))  # 모델 입력에 맞춤

#     # 3. numpy 변환 및 전처리
#     img_np = np.array(img).astype(np.float32) / 255.0
#     img_np = img_np.transpose(2, 0, 1)  # CHW
#     img_np = np.expand_dims(img_np, axis=0)  # (1, 3, 640, 640)

#     inputs = {"images": img_np}
#     outputs = session.run(None, inputs)
#     return outputs[0]
