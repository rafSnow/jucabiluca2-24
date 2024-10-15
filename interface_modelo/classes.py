from ultralytics import YOLO

# Função para carregar o modelo e obter os nomes das classes
def obter_nomes_classes(caminho_modelo, task='detect'):
    try:
        model = YOLO(caminho_modelo, task=task)
        class_names = model.names
        return class_names
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

# Caminho para o modelo treinado
caminho_modelo = r'C:\Users\Computador\Documents\DOCUMENTOSDIVERSOS\DocumentosFaculdade\6Periodo\APS\Codigo\jucabiluca2-24\best.pt'

# Obter os nomes das classes
nomes_classes = obter_nomes_classes(caminho_modelo)

# Se os nomes das classes não estiverem embutidos, forneça-os manualmente
if nomes_classes is None or all(isinstance(nome, int) for nome in nomes_classes):
    nomes_classes = [
        "pessoa", "bicicleta", "carro", "motocicleta", "avião",
        "ônibus", "trem", "caminhão", "barco", "sinal de trânsito",
        "hidrante", "placa de parada", "parquímetro", "banco", "pássaro",
        "gato", "cachorro", "cavalo", "ovelha", "vaca",
        "elefante", "urso", "zebra", "girafa", "mochila",
        "guarda-chuva", "bolsa", "gravata", "mala", "frisbee",
        "esquis", "snowboard", "bola de esportes", "pipa", "taco de beisebol",
        "luva de beisebol", "skate", "prancha de surf", "raquete de tênis", "garrafa",
        "taça de vinho", "copo", "garfo", "faca", "colher",
        "tigela", "banana", "maçã", "sanduíche", "laranja",
        "brócolis", "cenoura", "cachorro-quente", "pizza", "rosquinha",
        "bolo", "cadeira", "sofá", "planta em vaso", "cama",
        "mesa de jantar", "vaso sanitário", "televisão", "laptop", "mouse",
        "controle remoto", "teclado", "celular", "micro-ondas", "forno",
        "torradeira", "pia", "geladeira", "livro", "relógio",
        "vaso", "tesoura", "urso de pelúcia", "secador de cabelo", "escova de dentes"
    ]

if nomes_classes:
    print("Nomes das classes:")
    for nome in nomes_classes:
        print(nome)
else:
    print("Falha ao obter os nomes das classes.")