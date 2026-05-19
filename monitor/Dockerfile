FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Porta para o PostgreSQL (caso use Docker Compose)
EXPOSE 5432 8080

# Comando de inicialização
CMD ["python", "main.py"]
