FROM python:3.12-slim

WORKDIR /app

# 앱 파일 복사
COPY requirements.txt .
COPY main.py .
COPY labels.txt .

# 서비스 계정 키 복사 (로컬에서 함께 전달해야 함)
COPY key.json /app/key.json

# 의존성 설치 requirements.txt 파일을 사용함
RUN pip install -r requirements.txt




# 포트 노출 및 앱 실행
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
