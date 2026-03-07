import torch


print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Usando Device: {device}")
else:
    print("CUDA não detectada. Usando CPU.")
