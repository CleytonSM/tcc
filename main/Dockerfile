# Dockerfile para Raspberry Pi (ARM64)
FROM python:3.11-slim-bookworm

# Instalar dependências de sistema (OpenCV, GPIO, etc)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependências primeiro
COPY pyproject.toml .
RUN pip install .

# Copiar o código fonte e a lib compartilhada
COPY src/ ./src/
COPY ../../libs/shared/ ./libs/shared/

ENV PYTHONPATH="/app:/app/libs"

CMD ["python", "src/main.py"]
