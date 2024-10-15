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

# Função para carregar a imagem
def carregar_imagem(caminho):
    img = cv2.imread(caminho)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    return None

# Função para redimensionar a imagem para 640x640
def redimensionar_imagem(img):
    img_resized = img.resize((640, 640), Image.LANCZOS)
    return img_resized

# Função para recarregar o label com a imagem
def recarregar_label(label, img_tk):
    label.img_tk = img_tk  # Manter uma referência para evitar que a imagem seja coletada pelo garbage collector
    label.config(image=img_tk)

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

    img_final = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    return img_final


# Função para selecionar o arquivo
def selecionar_arquivo():
    caminho_imagem = filedialog.askopenfilename(
        title="Selecione a imagem para detecção", 
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    
    if caminho_imagem:
        # Atualizar a entrada de texto com o caminho do arquivo selecionado
        entry_caminho.delete(0, tk.END)
        entry_caminho.insert(0, caminho_imagem)

        # Carregar a imagem
        img = carregar_imagem(caminho_imagem)
        
        if img:
            # Redimensionar a imagem para caber no label
            img_resized = redimensionar_imagem(img)
            
            # Converter a imagem para o formato que o tkinter entende
            img_tk = ImageTk.PhotoImage(img_resized)
            
            # Recarregar e atualizar o label com a imagem
            label_img_original.config(image=img_tk, bg='black')
            label_img_original.image = img_tk  # Manter uma referência para evitar que a imagem seja coletada pelo garbage collector
        else:
            print("Erro ao carregar a imagem.")
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
        label_img_tratada.config(image=img_resultado_tk, bg='black')
        label_img_tratada.image = img_resultado_tk
    else:
        print("Nenhuma imagem selecionada.")

# Função para limpar o caminho e as imagens carregadas
def limpar():
    entry_caminho.delete(0, tk.END)
    label_img_original.config(image='')
    label_img_tratada.config(image='')

# Carregar o modelo treinado e obter os nomes das classes
caminho_modelo = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\6Periodo\APS\Codigo\jucabiluca2-24\interface_modelo\best.pt'
modelo, class_names = carregar_modelo(caminho_modelo)

# Criar a interface gráfica
root = tk.Tk()
root.title("Detecção de Objetos com YOLO")

# Centralizar a janela na tela
def centralizar_janela():
    root.update_idletasks()
    largura_janela = root.winfo_width()
    altura_janela = root.winfo_height()
    largura_tela = root.winfo_screenwidth()
    altura_tela = root.winfo_screenheight()
    pos_x = (largura_tela // 2) - (largura_janela // 2)
    pos_y = (altura_tela // 2) - (altura_janela // 2)
    root.geometry(f'{largura_janela}x{altura_janela}+{pos_x}+{pos_y}')

# Frame para conter os botões e o campo de entrada
frame_space = tk.Frame(root, bg="#f0f0f0")
frame_space.pack(side="top", pady=10)

# Campo para exibir o caminho do arquivo
entry_caminho = Entry(frame_space, width=50)
entry_caminho.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Botão para selecionar o arquivo
btn_selecionar = Button(frame_space, text="Selecionar Imagem", command=selecionar_arquivo, bg="#FFC107", fg="black", font=("Arial", 12, "bold"), borderwidth=2, relief="raised")
btn_selecionar.grid(row=0, column=0, padx=10, pady=10)

# Botão para iniciar a detecção
btn_detectar = Button(frame_space, text="Iniciar Detecção", command=iniciar_deteccao, bg="#FF5722", fg="white", font=("Arial", 12, "bold"), borderwidth=2, relief="raised")
btn_detectar.grid(row=0, column=2, padx=10, pady=10)

# Botão para limpar o caminho e as imagens carregadas
btn_limpar = Button(frame_space, text="Limpar Tela", command=limpar, bg="#795548", fg="white", font=("Arial", 12, "bold"), borderwidth=2, relief="raised")
btn_limpar.grid(row=0, column=3, padx=10, pady=10)

# Expansão dos widgets dentro do frame para que o espaço seja preenchido igualmente
frame_space.grid_columnconfigure(0, weight=1)
frame_space.grid_columnconfigure(1, weight=1)
frame_space.grid_columnconfigure(2, weight=1)
frame_space.grid_columnconfigure(3, weight=1)

# Frame para conter as imagens
frame_images = tk.Frame(root, bg="#f0f0f0")
frame_images.pack(expand=True, fill="both")

# Configurar o grid para o frame de imagens
frame_images.grid_rowconfigure(0, weight=1)
frame_images.grid_columnconfigure(0, weight=1)
frame_images.grid_columnconfigure(1, weight=1)


# Labels para exibir as imagens -- APARECE
label_img_original = tk.Label(frame_images, width=640, height=640, bg='#f0f0f0')
label_img_original.pack(side="left", padx=10, pady=10)

label_img_tratada = tk.Label(frame_images, width=640, height=640, bg='#f0f0f0')
label_img_tratada.pack(side="right", padx=10, pady=10)


# Centralizar a janela após a criação de todos os widgets
root.update_idletasks()
root.geometry('1350x750')  # Defina um tamanho inicial para a janela
centralizar_janela()

root.mainloop()
