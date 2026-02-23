# Dockerfile para Treinamento (GPU - AMD64)
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Instalar Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependências
COPY pyproject.toml .
RUN pip install .

# Copiar código e libs
COPY src/ ./src/
COPY ../../libs/shared/ ./libs/shared/
COPY ../../data/ ./data/

ENV PYTHONPATH="/app:/app/libs"

CMD ["python3", "src/train.py"]
