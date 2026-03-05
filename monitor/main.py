from ultralytics import YOLO

import cv2
import numpy as np
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

model = YOLO('best.pt')
video = cv2.VideoCapture("baby.mp4")

'''
Essas linhas de código são o "cérebro" do critério de decisão do DeepSORT. Elas definem como o rastreador decide se o "Bebê" que ele está vendo agora no frame 10 é o mesmo "Bebê" que ele viu no frame 9.

Aqui está o que cada parte faz:

1. max_dist = 0.4 (O Limite de Semelhança)
Esta variável controla o quão exigente o sistema é ao comparar a aparência (as features) de dois objetos.

O que é: O limite máximo permitido para considerar dois objetos como iguais.
Se a diferença for maior que esse valor, o sistema cria um novo ID.
0.0 significa que os objetos são idênticos.

2. nn_budget = None (A Memória do Rastreador)
Isso define quantos exemplos da aparência de cada objeto o sistema deve guardar na memória.

None: Significa que ele não tem limite fixo (ou usa o padrão da implementação).
Valor fixo (ex: 100): Se você colocar 100, ele guardará os últimos 100 frames da aparência do bebê. Isso ajuda em vídeos longos a não consumir toda a memória RAM.

3. nn_matching.NearestNeighborDistanceMetric(...) (A Ferramenta de Medição)
Aqui você está criando a "régua" que será usada para medir:

"cosine": Mede o ângulo entre os vetores de aparência (ideal para Re-ID). É mais robusto contra mudanças de iluminação.
"euclidean": Mede a distância em linha reta (a menor distância entre pontos). Usada quando não temos vetores de aparência (como o uso de np.zeros) ou quando as features são muito simples.

4. tracker = Tracker(metric) (O Gerenciador de Rastreamento)
Inicializa o objeto principal que gerencia os IDs, usa o Filtro de Kalman para prever posições e associa detecções usando a métrica definida acima.
'''
max_dist = 0.4
nn_budget = 100
metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_dist, nn_budget)
tracker = Tracker(metric)


class_names = ['baby', 'crib']

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    detections = []
    results = model(frame, verbose=False)
    for r in results:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf
        classes = boxes.cls

    
        for i in range(len(xyxy)):
            # 1. Corrigindo o formato tlwh (Top-Left, Width, Height)
            x1, y1, x2, y2 = xyxy[i]
            w = x2 - x1
            h = y2 - y1
            tlwh = [x1, y1, w, h]
            
            # 2. Criando uma feature dummy (ex: 128 zeros)
            # Se você não tiver o encoder de Re-ID, use isso:
            '''
            A feature no DeepSORT é o "DNA visual" ou "assinatura de aparência" do objeto.

            O que é a feature?
            É um vetor numérico (geralmente com 128 ou 512 números) que descreve as características visuais (cor, textura, formato) do que está dentro daquela caixa específica.

            Para que serve? Se o bebê passar atrás de um móvel (oclusão) e reaparecer, o DeepSORT compara a "feature" do bebê novo com a "feature" que ele tinha guardado. Mesmo que o bebê tenha mudado de posição, a feature visual será parecida, permitindo que ele mantenha o mesmo ID.
            De onde vem? Ela não vem do YOLO. Ela vem de uma segunda rede neural menor chamada Embedder ou Encoder (geralmente baseada em arquiteturas como a ResNet).
            '''
            feature = np.zeros(128) 
            
            # Se você já tiver um encoder, seria algo como:
            # patch = frame[int(y1):int(y2), int(x1):int(x2)]
            # feature = encoder(patch)
            # 3. Criando o objeto Detection
            detections.append(Detection(tlwh, confidences[i], feature, classes[i]))


     # 4. Atualizando o tracker com a lista de objetos Detection
    tracker.predict() # O DeepSORT original geralmente precisa do predict antes
    tracker.update(detections)
    
    # 5. Pegando os resultados do track
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
            
        # 1. Pegar as coordenadas (o track.to_tlbr() retorna [x1, y1, x2, y2])
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        
        class_id = int(track.class_id) 
        class_name = class_names[class_id] # Pega 'baby' ou 'crib'
        label = f"{class_name} #{track_id}"

        # 2. Desenhar o retângulo do objeto
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 3. Desenhar um fundo para o texto (opcional, para ficar mais bonito)
        cv2.rectangle(frame, (int(x1), int(y1) - 30), (int(x1) + 100, int(y1)), (255, 0, 0), -1)
        
        # 4. Escrever o ID do objeto
        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break