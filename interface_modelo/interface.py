import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

# Função para carregar o modelo treinado e obter as classes
def carregar_modelo(caminho_modelo):
    model = YOLO(caminho_modelo)
    class_names = model.names  # Obter os nomes das classes do modelo
    return model, class_names

# Função para carregar e processar a imagem
def carregar_imagem(caminho_imagem):
    img = Image.open(caminho_imagem)  # Abre a imagem como objeto PIL
    return img

# Função para redimensionar a imagem para 640x640
def redimensionar_imagem(img):
    img_resized = img.resize((640, 640), Image.LANCZOS)
    return img_resized

# Função para aplicar técnicas de pré-processamento mantendo a imagem colorida
def pre_processar_imagem(img):
    # Converter a imagem de PIL para NumPy
    img_np = np.array(img)

    # Convertendo a imagem para o formato BGR (usado pelo OpenCV)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Opções de pré-processamento:
    # Remova ou comente as técnicas uma a uma para testar

    # 1. Equalização de histograma (pode comentar essa linha para testar)
    # img_bgr[:, :, 0] = cv2.equalizeHist(img_bgr[:, :, 0])  # Canal B
    # img_bgr[:, :, 1] = cv2.equalizeHist(img_bgr[:, :, 1])  # Canal G
    # img_bgr[:, :, 2] = cv2.equalizeHist(img_bgr[:, :, 2])  # Canal R

    # 2. Suavização (pode comentar essa linha para testar)
    # img_suave = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    # # 3. Nitidez (pode comentar essa linha para testar)
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # img_sharp = cv2.filter2D(img_suave, -1, kernel)

    # Se remover a suavização e a nitidez, use apenas a imagem original em RGB
    img_final = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Converter de volta para objeto PIL
    img_final_pil = Image.fromarray(img_final)

    return img_final_pil


# Função para fazer a detecção com o modelo
def detectar_objetos(modelo, img):
    # Aplicar pré-processamento mantendo a imagem colorida
    img_preprocessada = pre_processar_imagem(img)

    # Redimensionar a imagem para 640x640
    img_resized = redimensionar_imagem(img_preprocessada)

    # Converter a imagem redimensionada em tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_resized).unsqueeze(0)  # Adiciona batch dimension

    # Passa a imagem pelo modelo para obter as previsões
    results = modelo(img_tensor, conf=0.25)  # Ajuste o limite de confiança conforme necessário

    return results, img_resized

# Função para exibir a imagem com as detecções
def exibir_resultados(img, results, class_names):
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Converter de PIL para OpenCV formato BGR

    # Definir cores para cada classe
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Dicionário para armazenar a contagem de detecções por classe
    detections_count = defaultdict(int)

    # Lista para armazenar as detecções
    detections_list = []

    # Loop para desenhar as detecções
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]  # Coordenadas da caixa delimitadora
            conf = detection.conf[0]  # Confiança da detecção
            cls = int(detection.cls[0])  # Classe da detecção
            class_name = class_names[cls]  # Nome da classe
            color = colors[cls % len(colors)]  # Cor da classe

            # Incrementar a contagem de detecções para a classe
            detections_count[class_name] += 1

            # Desenhar a caixa de detecção com linha pontilhada
            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(img_np, f'{class_name} {conf:.2%}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Adicionar as detecções à lista
    for class_name, count in detections_count.items():
        detections_list.append(f'{class_name}: {count} itens')

    # Desenhar a lista de detecções no canto da imagem
    for i, detection in enumerate(detections_list):
        cv2.putText(img_np, detection, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    # Exibir a imagem resultante
    plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Desativar os eixos
    plt.show()

# Programa principal
if __name__ == '__main__':
    # Carregar o modelo treinado e obter os nomes das classes
    caminho_modelo = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\6Periodo\APS\Codigo\jucabiluca2-24\interface_modelo\best.pt'
    modelo, class_names = carregar_modelo(caminho_modelo)

    # Abrir uma janela para selecionar a imagem do computador
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal do Tkinter
    caminho_imagem = filedialog.askopenfilename(title="Selecione a imagem para detecção", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if caminho_imagem:
        # Carregar a imagem selecionada
        imagem = carregar_imagem(caminho_imagem)

        # Fazer a detecção
        deteccoes, img_resized = detectar_objetos(modelo, imagem)

        # Exibir os resultados
        exibir_resultados(img_resized, deteccoes, class_names)
    else:
        print("Nenhuma imagem selecionada.")
