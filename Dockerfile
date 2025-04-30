FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY . /app
RUN chmod +x start.sh
EXPOSE 8501 8000 5000
CMD ["bash", "start.sh"]



