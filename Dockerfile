FROM python:3.12-slim
WORKDIR /app

# Use a faster Debian mirror (Aliyun for China)
RUN echo "deb http://mirrors.aliyun.com/debian bookworm main" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian bookworm-updates main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
        libpq-dev \
        gcc \
        g++ \
        libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
COPY . .
EXPOSE 8080
CMD ["python", "app.py"]
