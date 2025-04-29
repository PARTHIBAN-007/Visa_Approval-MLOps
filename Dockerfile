FROM python:3.12-slim
WORKDIR /app
RUN pip install uv 
COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt --system
COPY . .
CMD ["python", "main.py"]