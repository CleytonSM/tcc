import sys
import os

# Adicionando o caminho da lib compartilhada
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../libs')))

from shared.shared_logic import common_processing

def train():
    print("Iniciando Treinamento de ML (GPU)")
    data = "Conjunto de Dados de Imagens"
    processed = common_processing(data)
    print(f"Treinamento concluído. Modelo salvo em data/models/model.onnx")

if __name__ == "__main__":
    train()
