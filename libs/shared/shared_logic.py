def get_project_name():
    return "TCC Monorepo Project"

def common_processing(data):
    """
    Exemplo de lógica compartilhada (libs/shared).
    Poderia ser um pré-processamento de imagem ou um modelo Pydantic.
    """
    print(f"Processando dados: {data}")
    return data
