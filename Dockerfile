FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r backend/requirements.txt
EXPOSE 5050
CMD ["python", "backend/backend.py"] 