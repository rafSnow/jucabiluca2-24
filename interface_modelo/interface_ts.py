import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageTk
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, Label, Button, Entry, messagebox
from collections import defaultdict
import numpy as np
import cv2
import torch
from datetime import datetime

img_resultado = None

# Função para carregar o modelo treinado e obter as classes
# Retorna o modelo e os nomes das classes
def carregar_modelo(caminho_modelo):
    try:
        model = YOLO(caminho_modelo)
        class_names = model.names
        return model, class_names
    except AttributeError as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None, None

# Função para carregar a imagem
# Retorna a imagem carregada como um objeto PIL
def carregar_imagem(caminho):
    img = cv2.imread(caminho)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    return None

# Função para redimensionar a imagem para 640x640
# Retorna a imagem redimensionada
def redimensionar_imagem(img):
    img_resized = img.resize((640, 640), Image.LANCZOS)
    return img_resized

# Função para recarregar o label com a imagem
# Mantém uma referência para evitar que a imagem seja coletada pelo garbage collector
def recarregar_label(label, img_tk):
    label.img_tk = img_tk  # Manter uma referência para evitar que a imagem seja coletada pelo garbage collector
    label.config(image=img_tk)

# Função para aplicar pré-processamento na imagem
# Retorna a imagem pré-processada
def pre_processar_imagem(img):
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if var_equalizacao.get():
        # Equalização adaptativa de histograma (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_bgr[:, :, 0] = clahe.apply(img_bgr[:, :, 0])
        img_bgr[:, :, 1] = clahe.apply(img_bgr[:, :, 1])
        img_bgr[:, :, 2] = clahe.apply(img_bgr[:, :, 2])

    if var_suavizacao.get():
        # Suavização bilateral
        d = 9  # Diâmetro do pixel
        sigma_color = 75  # Filtro sigma no espaço de cor
        sigma_space = 75  # Filtro sigma no espaço de coordenadas
        img_bgr = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)

    if var_nitidez.get():
        # Nitidez com kernel de nitidez padrão
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_bgr = cv2.filter2D(img_bgr, -1, kernel)
        img_bgr = cv2.normalize(img_bgr, None, 0, 255, cv2.NORM_MINMAX)  # Normalizar para mitigar aumento de brilho

    if var_brilho.get():
        # Ajuste de brilho e contraste com valores fixos
        alpha = 2  # Contraste (1.0 - 3.0)
        beta = 20    # Brilho (0 - 100)
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    if var_normalizacao.get():
        # Normalização
        img_bgr = cv2.normalize(img_bgr, None, 0, 255, cv2.NORM_MINMAX)

    img_final = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_final_pil = Image.fromarray(img_final)

    return img_final_pil


# Função para fazer a detecção com o modelo
# Retorna as detecções e a imagem redimensionada
def detectar_objetos(modelo, img, conf_threshold):
    img_resized = img.resize((640, 640), Image.LANCZOS)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_resized).unsqueeze(0)

    results = modelo(img_tensor, conf=conf_threshold)  # Ajuste o limite de confiança conforme necessário

    return results, img_resized

# Função para exibir a imagem com as detecções
# Retorna a imagem com as caixas delimitadoras e as classes
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
        cv2.putText(img_np, detection, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    img_final = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    return img_final


# Função para aplicar zoom sobre a imagem no local do mouse
# A janela de zoom é exibida ao lado do cursor do mouse
def zoom_imagem(event, img, label_img, zoom_factor=2):
    # Obter o tamanho do label
    label_width = label_img.winfo_width()
    label_height = label_img.winfo_height()

    # Obter o tamanho da imagem
    img_width, img_height = img.size

    # Calcular a escala entre o label e a imagem
    scale_x = img_width / label_width
    scale_y = img_height / label_height

    # Coordenadas do mouse relativas à imagem
    x_mouse = int(event.x * scale_x)
    y_mouse = int(event.y * scale_y)

    # Tamanho do retângulo de zoom
    zoom_box_size = 60  # Ajuste conforme necessário

    # Limitar as coordenadas para não sair da imagem
    left = max(0, x_mouse - zoom_box_size)
    upper = max(0, y_mouse - zoom_box_size)
    right = min(img_width, x_mouse + zoom_box_size)
    lower = min(img_height, y_mouse + zoom_box_size)

    # Recortar a área ao redor do mouse
    img_crop = img.crop((left, upper, right, lower))

    # Redimensionar o recorte para simular o zoom
    img_zoom = img_crop.resize(
        ((right - left) * zoom_factor, (lower - upper) * zoom_factor),
        Image.LANCZOS
    )

    # Converter a imagem para o formato compatível com Tkinter
    img_zoom_tk = ImageTk.PhotoImage(img_zoom)

    # Exibir o zoom em uma janela pop-up próxima ao cursor
    if not hasattr(label_img, 'zoom_window') or not label_img.zoom_window.winfo_exists():
        # Criar uma nova janela Toplevel
        zoom_window = tk.Toplevel()
        zoom_window.overrideredirect(True)  # Remover decoração da janela
        zoom_window.attributes('-topmost', True)
        label_img.zoom_window = zoom_window
        zoom_label = tk.Label(zoom_window, image=img_zoom_tk)
        zoom_label.pack()
        label_img.zoom_label = zoom_label
    else:
        zoom_window = label_img.zoom_window
        zoom_label = label_img.zoom_label
        zoom_label.config(image=img_zoom_tk)

    # Manter referência para evitar coleta de lixo
    zoom_label.image = img_zoom_tk

    # Posicionar a janela de zoom próximo ao cursor
    x_root = event.x_root + 20
    y_root = event.y_root + 20
    zoom_window.geometry(f"+{x_root}+{y_root}")

# Função para fechar a janela de zoom ao sair do label
# Verifica se a janela de zoom existe antes de tentar fechar
def close_zoom(event, label_img):
    if hasattr(label_img, 'zoom_window') and label_img.zoom_window.winfo_exists():
        label_img.zoom_window.destroy()

# Modificar a função selecionar_arquivo para adicionar o bind dos eventos
# de zoom para a imagem original
def selecionar_arquivo():

    entry_caminho.delete(0, tk.END)
    label_img_original.config(image='', bg='#f0f0f0')  # Restaurar a cor de fundo para branco
    label_img_tratada.config(image='', bg='#f0f0f0')  # Restaurar a cor de fundo para branco
    
    # Desmarcar todas as checkboxes
    var_equalizacao.set(False)
    var_suavizacao.set(False)
    var_nitidez.set(False)
    var_brilho.set(False)
    var_normalizacao.set(False)
    
    # Limpar o campo de entrada de confiança
    entry_conf.delete(0, tk.END)
    entry_conf.insert(0, "25")  # Valor padrão de confiança

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

            # Associar eventos para zoom
            label_img_original.bind("<Motion>", lambda event: zoom_imagem(event, img_resized, label_img_original))
            label_img_original.bind("<Leave>", lambda event: close_zoom(event, label_img_original))
            label_titulo_original.config(text="Imagem Original")
        else:
            print("Erro ao carregar a imagem.")
    else:
        print("Nenhuma imagem selecionada.")


# Função para iniciar a detecção
def iniciar_deteccao():
    global img_resultado
    caminho_imagem = entry_caminho.get()
    if caminho_imagem:
        img = carregar_imagem(caminho_imagem)
        if img:
            # Aplicar pré-processamento na imagem
            img_preprocessada = pre_processar_imagem(img)
            
            # Obter o valor de confiança do campo de entrada
            conf_threshold = float(entry_conf.get()) / 100
            
            # Realizar a detecção de objetos
            deteccoes, img_resized = detectar_objetos(modelo, img_preprocessada, conf_threshold)
            
            # Exibir os resultados
            img_resultado = exibir_resultados(img_resized, deteccoes, class_names)
            img_resultado_tk = ImageTk.PhotoImage(img_resultado) # Converta a imagem para o formato que o tkinter entende
            label_img_tratada.config(image=img_resultado_tk, bg='black')
            label_img_tratada.image = img_resultado_tk

            # Adicionar eventos de zoom também para a imagem tratada
            label_img_tratada.bind("<Motion>", lambda event: zoom_imagem(event, img_resultado, label_img_tratada))
            label_img_tratada.bind("<Leave>", lambda event: close_zoom(event, label_img_tratada))

            # Enable the save button
            btn_salvar_imagem.config(state=tk.NORMAL)

            label_titulo_tratada.config(text="Imagem Com Detecções")

        else:
            print("Erro ao carregar a imagem.")
    else:
        print("Nenhuma imagem selecionada.")

# Função para salvar a imagem exibida
def salvar_imagem():
    global img_resultado  # Declare as global to access the global variable
    if img_resultado:
        # Get current date and time
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        default_filename = f"ImagemTratada_{timestamp}.png"
        
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_filename, filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            img_resultado.save(file_path)
            messagebox.showinfo("Imagem Salva", f"Imagem salva em {file_path}")
    else:
        messagebox.showwarning("Nenhuma Imagem", "Nenhuma imagem para salvar.")


# Função para iniciar detecção ao abrir aplicativo
def start_imagem():
    caminho_img = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\6Periodo\APS\Codigo\jucabiluca2-24\interface_modelo\imagens_teste\canteiro-de-obras-1-1-1024x576.jpg'
    img = carregar_imagem(caminho_img)

    # Redimensionar a imagem para caber no label
    img_resized = redimensionar_imagem(img)
            
    # Converter a imagem para o formato que o tkinter entende
    img_tk = ImageTk.PhotoImage(img_resized)
            
    # Recarregar e atualizar o label com a imagem
    label_img_original.config(image=img_tk, bg='black')
    label_img_original.image = img_tk  # Manter uma referência para evitar que a imagem seja coletada pelo garbage collector

    # Associar eventos para zoom
    label_img_original.bind("<Motion>", lambda event: zoom_imagem(event, img_resized, label_img_original))
    label_img_original.bind("<Leave>", lambda event: close_zoom(event, label_img_original))
    
    # Aplicar pré-processamento na imagem
    img_preprocessada = pre_processar_imagem(img)
            
    # Obter o valor de confiança do campo de entrada
    conf_threshold = float(entry_conf.get()) / 100
            
    # Realizar a detecção de objetos
    deteccoes, img_resized = detectar_objetos(modelo, img_preprocessada, conf_threshold)
            
    # Exibir os resultados
    img_resultado = exibir_resultados(img_resized, deteccoes, class_names)
    img_resultado_tk = ImageTk.PhotoImage(img_resultado)
    label_img_tratada.config(image=img_resultado_tk, bg='black')
    label_img_tratada.image = img_resultado_tk

    # Adicionar eventos de zoom também para a imagem tratada
    label_img_tratada.bind("<Motion>", lambda event: zoom_imagem(event, img_resultado, label_img_tratada))
    label_img_tratada.bind("<Leave>", lambda event: close_zoom(event, label_img_tratada))


# Função para limpar o caminho e as imagens carregadas
def limpar():
    entry_caminho.delete(0, tk.END)
    label_img_original.config(image='', bg='#f0f0f0')  # Restaurar a cor de fundo para branco
    label_img_original.unbind("<Motion>")
    label_img_original.unbind("<Leave>")
    label_img_original.image = None  # Limpar a imagem para liberar memória

    label_img_tratada.config(image='', bg='#f0f0f0')  # Restaurar a cor de fundo para branco
    label_img_tratada.unbind("<Motion>")
    label_img_tratada.unbind("<Leave>")
    label_img_tratada.image = None  # Limpar a imagem para liberar memória

    label_titulo_original.config(text="")
    label_titulo_tratada.config(text="")
    
    # Desmarcar todas as checkboxes
    var_equalizacao.set(False)
    var_suavizacao.set(False)
    var_nitidez.set(False)
    var_brilho.set(False)
    var_normalizacao.set(False)
    
    # Limpar o campo de entrada de confiança
    entry_conf.delete(0, tk.END)
    entry_conf.insert(0, "25")  # Valor padrão de confiança

# Função para exibir a descrição ao passar o mouse
def mostrar_descricao(event, descricao):
    tooltip = tk.Toplevel()
    tooltip.wm_overrideredirect(True)
    tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
    tooltip.attributes('-topmost', True)  # Manter a janela em primeiro plano
    label = tk.Label(tooltip, text=descricao, background="yellow", relief="solid", borderwidth=1, font=("Arial", 10))
    label.pack()
    event.widget.tooltip = tooltip

# Função para esconder a descrição ao sair do mouse
def esconder_descricao(event):
    if hasattr(event.widget, 'tooltip'):
        event.widget.tooltip.destroy()
        del event.widget.tooltip

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

# Função para centralizar a janela na tela do usuário
def centralizar_janela_inicial(janela):
    janela.update_idletasks()  # Garante que o tamanho final da janela seja calculado
    largura_janela = janela.winfo_width()
    altura_janela = janela.winfo_height()
    largura_tela = janela.winfo_screenwidth()
    altura_tela = janela.winfo_screenheight()
    
    pos_x = (largura_tela // 2) - (largura_janela // 2)
    pos_y = (altura_tela // 2) - (altura_janela // 2)
    
    janela.geometry(f'{largura_janela}x{altura_janela}+{pos_x}+{pos_y}')

# Função para exibir a janela de ajuda estilizada
def exibir_ajuda():
    ajuda_janela = tk.Toplevel(root)
    ajuda_janela.title("Ajuda")
    ajuda_janela.geometry("500x650")
    ajuda_janela.configure(bg="#2c3e50")  # Fundo escuro
    ajuda_janela.attributes('-topmost', True)  # Manter a janela em primeiro plano
    
    ajuda_texto = """
    Bem-vindo ao aplicativo de detecção de objetos!

    Funcionalidades:\n
    - Selecionar Imagem: Permite selecionar uma imagem do seu computador.
    - Iniciar Detecção: Inicia o processo de detecção de objetos na imagem selecionada.
    - Limpar Tela: Limpa a imagem carregada e os resultados da detecção.
    - Salvar Imagem: Salva a imagem com os resultados da detecção.

    O que esperar:\n
    - O aplicativo irá carregar a imagem selecionada e aplicar técnicas de detecção de objetos.
    - Os resultados serão exibidos na tela.
    - Você pode ajustar o valor de confiança para a detecção.

                    Aproveite o aplicativo!
    """
    
    label_ajuda = tk.Label(ajuda_janela, text=ajuda_texto, justify="left", padx=10, pady=10, bg="#34495e", fg="white", font=("Arial", 14, "bold"), wraplength=400)
    label_ajuda.pack(expand=True, fill="both")
    centralizar_janela_inicial(ajuda_janela)

# Carregar o modelo treinado e obter os nomes das classes
caminho_modelo = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\6Periodo\APS\Codigo\jucabiluca2-24\best.torchscript'
modelo, class_names = carregar_modelo(caminho_modelo)
if modelo is None:
    print("Falha ao carregar o modelo. Verifique a compatibilidade da versão da biblioteca 'ultralytics'.")
else:
    print("Modelo carregado com sucesso.")

# Criar a interface gráfica
root = tk.Tk()
root.title("Detecção de Objetos com YOLO")
root.attributes('-topmost', True)  # Manter a janela em primeiro plano


# Frame para conter os botões e o campo de entrada
frame_space = tk.Frame(root, bg="#f0f0f0")
frame_space.pack(side="top", pady=10)

# Tamanho padrão para os botões
button_width = 20
button_height = 2

# Botão para selecionar o arquivo
btn_selecionar = tk.Button(frame_space, text="Selecionar Imagem", command=selecionar_arquivo, bg="#FFC107", fg="black", font=("Arial", 12, "bold"), borderwidth=2, relief="raised", width=button_width, height=button_height)
btn_selecionar.grid(row=0, column=0, padx=10, pady=10)

# Botão para iniciar a detecção
btn_detectar = tk.Button(frame_space, text="Iniciar Detecção", command=iniciar_deteccao, bg="#216e0f", fg="black", font=("Arial", 12, "bold"), borderwidth=2, relief="raised", width=button_width, height=button_height)
btn_detectar.grid(row=0, column=1, padx=10, pady=10)

# Botão para limpar o caminho e as imagens carregadas
btn_limpar = tk.Button(frame_space, text="Limpar Tela", command=limpar, bg="#8f0303", fg="black", font=("Arial", 12, "bold"), borderwidth=2, relief="raised", width=button_width, height=button_height)
btn_limpar.grid(row=0, column=2, padx=10, pady=10)

# Botão para salvar a imagem
btn_salvar_imagem = tk.Button(frame_space, text="Salvar Imagem", command=salvar_imagem, bg="#FFC107", fg="black", font=("Arial", 12, "bold"), borderwidth=2, relief="raised", width=button_width, height=button_height, state=tk.DISABLED)
btn_salvar_imagem.grid(row=0, column=3, padx=10, pady=10)

# Adicionar colunas vazias para centralizar os elementos
frame_space.columnconfigure(0, weight=1)
frame_space.columnconfigure(5, weight=1)

# Variável oculta para o campo de entrada do caminho do arquivo
entry_caminho = tk.Entry(frame_space, width=50, font=("Arial", 10, "italic"))

# Campo para inserir o valor de confiança
label_conf = tk.Label(frame_space, text="Confiança das detecções (%)", bg="#f0f0f0", font=("Arial", 12, "bold"))
label_conf.grid(row=2, column=1, padx=5, pady=5, sticky="e")
entry_conf = tk.Entry(frame_space, width=10, font=("Arial", 10, "italic"))
entry_conf.grid(row=2, column=2, padx=5, pady=5, sticky="w")
entry_conf.insert(0, "25")  # Valor padrão de confiança

# Expansão dos widgets dentro do frame para que o espaço seja preenchido igualmente
frame_space.grid_columnconfigure(1, weight=1)
frame_space.grid_columnconfigure(2, weight=1)
frame_space.grid_columnconfigure(3, weight=1)
frame_space.grid_columnconfigure(4, weight=1)

# Frame para conter os checkboxes de preprocessamento
frame_preprocessamento = tk.Frame(root, bg="#f0f0f0")
frame_preprocessamento.pack(side="top", pady=10)

# Variáveis para armazenar o estado dos checkboxes
var_equalizacao = tk.BooleanVar()
var_suavizacao = tk.BooleanVar()
var_nitidez = tk.BooleanVar()
var_brilho = tk.BooleanVar()
var_normalizacao = tk.BooleanVar()

# Checkboxes para as técnicas de preprocessamento
chk_equalizacao = tk.Checkbutton(frame_preprocessamento, text="Equalização de Histograma", variable=var_equalizacao, bg="#f0f0f0", font=("Arial", 12, "bold"))
chk_equalizacao.pack(side="left", padx=10, pady=10)
chk_equalizacao.bind("<Enter>", lambda event: mostrar_descricao(event, "Equalização de Histograma: Melhora o contraste da imagem ao redistribuir os níveis de intensidade. \nIsso pode ajudar na detecção de objetos em áreas com pouca iluminação ou sombras."))
chk_equalizacao.bind("<Leave>", esconder_descricao)

chk_suavizacao = tk.Checkbutton(frame_preprocessamento, text="Suavização", variable=var_suavizacao, bg="#f0f0f0", font=("Arial", 12, "bold"))
chk_suavizacao.pack(side="left", padx=10, pady=10)
chk_suavizacao.bind("<Enter>", lambda event: mostrar_descricao(event, "Suavização: Reduz o ruído da imagem aplicando um filtro de desfoque. \nIsso pode melhorar a detecção de objetos ao eliminar detalhes irrelevantes, mas pode suavizar bordas importantes."))
chk_suavizacao.bind("<Leave>", esconder_descricao)

chk_nitidez = tk.Checkbutton(frame_preprocessamento, text="Nitidez", variable=var_nitidez, bg="#f0f0f0", font=("Arial", 12, "bold"))
chk_nitidez.pack(side="left", padx=10, pady=10)
chk_nitidez.bind("<Enter>", lambda event: mostrar_descricao(event, "Nitidez: Aumenta a nitidez da imagem ao realçar bordas e detalhes. \nIsso pode melhorar a detecção de objetos ao tornar os contornos mais definidos, mas pode aumentar o ruído."))
chk_nitidez.bind("<Leave>", esconder_descricao)

chk_brilho = tk.Checkbutton(frame_preprocessamento, text="Ajuste de Brilho", variable=var_brilho, bg="#f0f0f0", font=("Arial", 12, "bold"))
chk_brilho.pack(side="left", padx=10, pady=10)
chk_brilho.bind("<Enter>", lambda event: mostrar_descricao(event, "Ajuste de Brilho: Ajusta o brilho da imagem para torná-la mais clara ou mais escura. \nIsso pode ajudar na detecção de objetos em condições de iluminação inadequadas, mas pode saturar a imagem se usado em excesso."))
chk_brilho.bind("<Leave>", esconder_descricao)

chk_normalizacao = tk.Checkbutton(frame_preprocessamento, text="Normalização", variable=var_normalizacao, bg="#f0f0f0", font=("Arial", 12, "bold"))
chk_normalizacao.pack(side="left", padx=10, pady=10)
chk_normalizacao.bind("<Enter>", lambda event: mostrar_descricao(event, "Normalização: Normaliza os valores dos pixels da imagem para um intervalo padrão. \nIsso pode melhorar a detecção de objetos ao garantir que a imagem tenha uma distribuição de\n intensidade consistente, facilitando a análise pelo modelo."))
chk_normalizacao.bind("<Leave>", esconder_descricao)

# Frame para conter as imagens
frame_images = tk.Frame(root, bg="#f0f0f0")
frame_images.pack(expand=True, fill="both")

# Frame para a imagem original
frame_img_original = tk.Frame(frame_images, bg="#f0f0f0")
frame_img_original.pack(side="left", padx=10, pady=10)

# Frame para a imagem tratada
frame_img_tratada = tk.Frame(frame_images, bg="#f0f0f0")
frame_img_tratada.pack(side="right", padx=10, pady=10)

# Labels para os títulos das imagens
label_titulo_original = tk.Label(frame_img_original, text="Imagem Original", bg='#f0f0f0', font=("Arial", 14, "bold"))
label_titulo_original.pack()

label_titulo_tratada = tk.Label(frame_img_tratada, text="Imagem Com Detecções", bg='#f0f0f0', font=("Arial", 14, "bold"))
label_titulo_tratada.pack()

# Labels para exibir as imagens
label_img_original = tk.Label(frame_img_original, width=640, height=640, bg='#f0f0f0')
label_img_original.pack()

label_img_tratada = tk.Label(frame_img_tratada, width=640, height=640, bg='#f0f0f0')
label_img_tratada.pack()

# Centralizar a janela após a criação de todos os widgets
root.update_idletasks()
root.geometry('1350x950')  # Defina um tamanho inicial para a janela
centralizar_janela()

# # Adicionar um atraso de 1 segundo antes de executar a função start_imagem
# root.after(2000, start_imagem)
start_imagem()

# Exibir a janela de ajuda ao iniciar o programa
root.after(1000, exibir_ajuda)

root.mainloop()