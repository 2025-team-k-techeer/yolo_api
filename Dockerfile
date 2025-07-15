FROM python:3.10-slim-bookworm

WORKDIR /app

COPY . .

# 서비스 계정 키는 런타임에 볼륨으로 전달해야 함
# 환경 변수는 런타임에 설정해야 함

# 의존성 설치 requirements.txt 파일을 사용함
RUN pip install -r requirements.txt


# 포트 노출 및 앱 실행
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
