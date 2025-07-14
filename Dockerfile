FROM python:3.12-slim

WORKDIR /app

# 앱 파일 복사
COPY uv.lock .
COPY pyproject.toml .
COPY main.py .


# 서비스 계정 키 복사 (로컬에서 함께 전달해야 함)
COPY key.json /app/key.json

# 의존성 설치
RUN pip install uv

RUN uv lock 

# gcloud 설치
RUN apt-get update && \
	apt-get install -y curl gnupg ca-certificates && \
	echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
	curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
	apt-get update && apt-get install -y google-cloud-sdk

# 서비스 계정 인증 및 모델 다운로드
RUN gcloud auth activate-service-account --key-file=/app/key.json && \
	gcloud config set project latex-server-410907 && \
	gsutil cp gs://k-ai-model-bucket/yolo/yolov8n.onnx /app/model.onnx

# 포트 노출 및 앱 실행
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
