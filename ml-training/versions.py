import torch
import torch_directml

print(f"PyTorch Version: {torch.__version__}")
print(f"DirectML Available: {torch_directml.is_available()}")

if torch_directml.is_available():
    device = torch_directml.device()
    print(f"Usando Device: {device}")
else:
    print("GPU AMD não detectada via DirectML. Usando CPU.")
