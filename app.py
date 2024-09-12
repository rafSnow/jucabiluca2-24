from flask import Flask, request, send_file, jsonify, Response
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Carrega o modelo
model = YOLO("best.pt")
#qqq
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ler o arquivo da Imagem
    image = Image.open(file.stream)

    # Converte a imagem para o formato YOLOv8 para processamento (numpy array)
    image_np = np.array(image)

    # Inferencia do Resultado
    results = model(image_np)

    # Processar o resultado e salvara imagem
    result_image = results[0].plot()  
    result_image_pil = Image.fromarray(result_image) 

    # Converter Imagem PIL para bytes
    img_io = io.BytesIO()
    result_image_pil.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)

    return Response(img_io, mimetype='image/jpeg')

@app.route('/')
def index():
    return '''
        <!doctype html>
        <html>
        <head>
            <title>Upload Image</title>
        </head>
        <body>
            <h1>Upload Image for Processing</h1>
            <form action="/process-image" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Upload">
            </form>
        </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)

# from ultralytics import YOLO
# import cv2
# import math 

# # iniciar webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# # modelo
# model = YOLO("best.pt")

# # classes de objetos
# classNames = ["pessoa", "bicicleta", "carro", "moto", "avião", "ônibus", "trem", "caminhão", "barco",
#               "sinal de trânsito", "hidrante", "placa de pare", "parquímetro", "banco", "pássaro", "gato",
#               "cachorro", "cavalo", "ovelha", "vaca", "elefante", "urso", "zebra", "girafa", "mochila", "guarda-chuva",
#               "bolsa", "gravata", "mala", "frisbee", "esquis", "snowboard", "bola esportiva", "pipa", "bastão de beisebol",
#               "luva de beisebol", "skate", "prancha de surfe", "raquete de tênis", "garrafa", "copo de vinho", "xícara",
#               "garfo", "faca", "colher", "tigela", "banana", "maçã", "sanduíche", "laranja", "brócolis",
#               "cenoura", "cachorro-quente", "pizza", "donut", "bolo", "cadeira", "sofá", "planta em vaso", "cama",
#               "mesa de jantar", "vaso sanitário", "monitor de TV", "laptop", "mouse", "controle remoto", "teclado", "celular",
#               "micro-ondas", "forno", "torradeira", "pia", "geladeira", "livro", "relógio", "vaso", "tesoura",
#               "ursinho de pelúcia", "secador de cabelo", "escova de dentes", "escavadora", "luvas", "capacete", "máscara"]

# while True:
#     sucesso, img = cap.read()
#     if not sucesso:
#         print("Falha ao capturar o quadro")
#         break

#     resultados = model(img, stream=True)

#     # coordenadas
#     for r in resultados:
#         caixas = r.boxes

#         for caixa in caixas:
#             # caixa delimitadora
#             x1, y1, x2, y2 = caixa.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # converter para valores inteiros

#             # desenhar a caixa na imagem
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#             # confiança
#             confianca = math.ceil((caixa.conf[0]*100))/100
#             print("Confiança --->", confianca)

#             # nome da classe
#             cls = int(caixa.cls[0])
#             print("Nome da classe -->", classNames[cls])

#             # detalhes do objeto
#             org = [x1, y1]
#             fonte = cv2.FONT_HERSHEY_SIMPLEX
#             escalaFonte = 1
#             cor = (255, 0, 0)
#             espessura = 2

#             cv2.putText(img, classNames[cls], org, fonte, escalaFonte, cor, espessura)

#     if img is not None and img.size > 0:
#         cv2.imshow('Webcam', img)
#     else:
#         print("Quadro vazio recebido")

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()