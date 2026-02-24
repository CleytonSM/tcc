import sys
import os

# Adicionando o caminho da lib compartilhada (em desenvolvimento)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../libs')))

from shared.shared_logic import common_processing

def main():
    print("Iniciando Aplicativo Embarcado (Raspberry Pi)")
    data = "Dados do Sensor"
    processed = common_processing(data)
    print(f"Resultado: {processed}")

if __name__ == "__main__":
    main()
