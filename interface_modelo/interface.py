import numpy as np
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

# Função para aplicar pré-processamento na imagem
def pre_processar_imagem(img):
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Equalização de histograma
    img_bgr[:, :, 0] = cv2.equalizeHist(img_bgr[:, :, 0])
    img_bgr[:, :, 1] = cv2.equalizeHist(img_bgr[:, :, 1])
    img_bgr[:, :, 2] = cv2.equalizeHist(img_bgr[:, :, 2])

    # Suavização
    img_suave = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    # Nitidez
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_suave, -1, kernel)

    # Ajuste de brilho e contraste
    alpha = 1.2  # Contraste
    beta = 50    # Brilho
    img_bright_contrast = cv2.convertScaleAbs(img_sharp, alpha=alpha, beta=beta)

    # Normalização
    img_normalized = cv2.normalize(img_bright_contrast, None, 0, 255, cv2.NORM_MINMAX)

    img_final = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
    img_final_pil = Image.fromarray(img_final)

    return img_final_pil

# Função para redimensionar a imagem para 640x640
def redimensionar_imagem(img):
    img_resized = img.resize((640, 640), Image.LANCZOS)
    return img_resized

# Função para fazer a detecção com o modelo
def detectar_objetos(modelo, img):
    img_preprocessada = pre_processar_imagem(img)
    img_resized = redimensionar_imagem(img_preprocessada)

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_resized).unsqueeze(0)

    results = modelo(img_tensor, conf=0.25)  # Ajuste o limite de confiança conforme necessário

    return results, img_resized

# Função para exibir a imagem com as detecções
def exibir_resultados(img, results, class_names):
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    detections_count = defaultdict(int)
    detections_list = []

    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            conf = detection.conf[0]
            cls = int(detection.cls[0])
            class_name = class_names[cls]
            color = colors[cls % len(colors)]

            detections_count[class_name] += 1

            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(img_np, f'{class_name} {conf:.2%}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for class_name, count in detections_count.items():
        detections_list.append(f'{class_name}: {count} itens')

    for i, detection in enumerate(detections_list):
        cv2.putText(img_np, detection, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Programa principal
if __name__ == '__main__':
    caminho_modelo = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\6Periodo\APS\Codigo\jucabiluca2-24\interface_modelo\best.pt'
    modelo, class_names = carregar_modelo(caminho_modelo)

    root = tk.Tk()
    root.withdraw()
    caminho_imagem = filedialog.askopenfilename(title="Selecione a imagem para detecção", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if caminho_imagem:
        imagem = carregar_imagem(caminho_imagem)
        deteccoes, img_resized = detectar_objetos(modelo, imagem)
        exibir_resultados(img_resized, deteccoes, class_names)
    else:
        print("Nenhuma imagem selecionada.")