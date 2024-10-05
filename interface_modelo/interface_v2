import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageTk
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, Label, Button, Entry
from collections import defaultdict
import numpy as np
import cv2

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
    alpha = 1  # Contraste
    beta = 5    # Brilho
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

    results = modelo(img_tensor, conf=0.35)  # Ajuste o limite de confiança conforme necessário

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
        detections_list.append(f'{class_name}: {count} item (ns)')

    for i, detection in enumerate(detections_list):
        cv2.putText(img_np, detection, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        # Para determinar a cor da fonte do texto em amarelo ou branco, dependendo da cor de fundo
        # cv2.putText(img_np, detection, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if np.mean(img_np[:50, :50]) < 127 else (0, 0, 0), 2, lineType=cv2.LINE_AA)
    img_final = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    return img_final

# Função para selecionar o arquivo
def selecionar_arquivo():
    caminho_imagem = filedialog.askopenfilename(title="Selecione a imagem para detecção", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if caminho_imagem:
        entry_caminho.delete(0, tk.END)
        entry_caminho.insert(0, caminho_imagem)
        img = carregar_imagem(caminho_imagem)
        img_resized = redimensionar_imagem(img)
        img_tk = ImageTk.PhotoImage(img_resized)
        label_img_original.config(image=img_tk)
        label_img_original.image = img_tk
    else:
        print("Nenhuma imagem selecionada.")

# Função para iniciar a detecção
def iniciar_deteccao():
    caminho_imagem = entry_caminho.get()
    if caminho_imagem:
        img = carregar_imagem(caminho_imagem)
        deteccoes, img_resized = detectar_objetos(modelo, img)
        img_resultado = exibir_resultados(img_resized, deteccoes, class_names)
        img_resultado_tk = ImageTk.PhotoImage(img_resultado)
        label_img_tratada.config(image=img_resultado_tk)
        label_img_tratada.image = img_resultado_tk
    else:
        print("Nenhuma imagem selecionada.")

# Carregar o modelo treinado e obter os nomes das classes
caminho_modelo = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\6Periodo\APS\Codigo\jucabiluca2-24\interface_modelo\best.pt'
modelo, class_names = carregar_modelo(caminho_modelo)

# Criar a interface gráfica
root = tk.Tk()
root.title("Detecção de Objetos com YOLO")

# Campo para exibir o caminho do arquivo
entry_caminho = Entry(root, width=50)
entry_caminho.grid(row=0, column=1, padx=10, pady=10)

# Botão para selecionar o arquivo
btn_selecionar = Button(root, text="Selecionar Arquivo", command=selecionar_arquivo)
btn_selecionar.grid(row=0, column=0, padx=10, pady=10)

# Botão para iniciar a detecção
btn_detectar = Button(root, text="Iniciar Detecção", command=iniciar_deteccao)
btn_detectar.grid(row=0, column=2, padx=10, pady=10)

# Labels para exibir as imagens
label_img_original = Label(root)
label_img_original.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

label_img_tratada = Label(root)
label_img_tratada.grid(row=1, column=2, columnspan=2, padx=10, pady=10)

root.mainloop()