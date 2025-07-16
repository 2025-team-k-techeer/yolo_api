import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from io import BytesIO


def get_clip_session():
    return onnxruntime.InferenceSession(
        "models/static_model.onnx",  # 우리가 만든 모델 경로
        providers=["CPUExecutionProvider"],
    )


def preprocess_image(image: Image.Image) -> np.ndarray:
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4815, 0.4578, 0.4082],
                std=[0.2686, 0.2613, 0.2758],
            ),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor.numpy()


def get_clip_embedding_from_uploadfile(upload_file, session):
    image_bytes = BytesIO(upload_file.file.read())
    img = Image.open(image_bytes).convert("RGB")
    input_tensor = preprocess_image(img)

    # 입력 이름은 반드시 모델에 따라 맞춰야 함
    inputs = {"pixel_values": input_tensor}

    outputs = session.run(None, inputs)
    return outputs[0]  # shape: (1, 50, 768)
