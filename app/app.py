import base64
import io
import os
import sys

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "pipeline"))

from caminhos import MODELOS
from inferencia_singleview import EnsembleSingleView

MODELOS_DIR = os.path.join(MODELOS, "modelos_final_singleview")
YOLO_PATH = os.path.join(MODELOS, "yolo11n-seg.pt")
FRONTEND_DIST = os.path.join(SCRIPT_DIR, "static", "dist")


CLASSES_ANIMAL = [16, 17, 18, 19, 20]
AREA_MINIMA_PX = 1000  
FAIXA_CONFIAVEL_KG = (90.0, 248.0)
MARGEM_P80_KG = 40.4

FAIXAS = [
    (0, 250, "Leve (< 250 kg)"),
    (250, 400, "Médio (250–400 kg)"),
    (400, 9999, "Pesado (> 400 kg)"),
]

app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_cache = None
yolo_cache = None


def carregar_modelo():
    global modelo_cache
    if modelo_cache is None:
        modelo_cache = EnsembleSingleView(MODELOS_DIR, device)
    return modelo_cache


def carregar_yolo():
    global yolo_cache
    if yolo_cache is None:
        from ultralytics import YOLO
        yolo_cache = YOLO(YOLO_PATH)
    return yolo_cache


def segmentar(img_rgb):
    """Máscara (HxW bool) do maior animal detectado, ou None se não houver nenhum.

    Fica só com a maior porque o modelo estima um animal: unir as máscaras de dois
    bois somaria as áreas e alteraria as medidas relativas ao quadro.
    """
    h, w = img_rgb.shape[:2]
    # em array numpy o YOLO espera BGR
    resultado = carregar_yolo()(img_rgb[..., ::-1].copy(), conf=0.25, classes=CLASSES_ANIMAL, verbose=False)[0]
    if resultado.masks is None:
        return None

    maior = None
    for m in resultado.masks.data.cpu().numpy():
        mascara = cv2.resize((m > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if maior is None or mascara.sum() > maior.sum():
            maior = mascara
    if maior is None or maior.sum() < AREA_MINIMA_PX:
        return None
    return maior


def img_to_base64(img_rgb):
    img = Image.fromarray(img_rgb)
    img.thumbnail((640, 480))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def classificar_faixa(peso_kg):
    for baixo, alto, nome in FAIXAS:
        if baixo <= peso_kg < alto:
            return nome
    return "Indefinida"


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    arquivo = request.files.get("imagem")
    if arquivo is None or arquivo.filename == "":
        return jsonify({"erro": "Nenhuma imagem enviada."}), 400

    sexo = request.form.get("sexo")
    if sexo not in ("F", "M"):
        return jsonify({"erro": "Informe o sexo do animal."}), 400

    try:
        # celular costuma salvar a foto deitada com uma tag EXIF de rotação
        img_rgb = np.array(ImageOps.exif_transpose(Image.open(arquivo.stream)).convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify({"erro": "Não foi possível abrir essa imagem. Envie um JPG ou PNG."}), 400

    mascara = segmentar(img_rgb)
    if mascara is None:
        return jsonify({"erro": "Não encontramos um boi nessa foto. Tire outra com o animal inteiro e visível."}), 422

    peso_kg = carregar_modelo().prever(img_rgb, mascara, 1.0 if sexo == "M" else 0.0)["peso_kg"]

    segmentada = img_rgb.copy()
    segmentada[~mascara] = 0
    return jsonify({
        "peso_kg": round(peso_kg, 1),
        "faixa": classificar_faixa(peso_kg),
        "margem_kg": MARGEM_P80_KG,
        "treino_min": FAIXA_CONFIAVEL_KG[0],
        "treino_max": FAIXA_CONFIAVEL_KG[1],
        "img_original": img_to_base64(img_rgb),
        "img_segmentada": img_to_base64(segmentada),
    })


if __name__ == "__main__":
    print(f"Dispositivo: {device}")
    print("Pré-carregando os 5 modelos EfficientNetV2-S...")
    carregar_modelo()
    print("Pré-carregando YOLO11n-seg...")
    carregar_yolo()
    print("\nServidor rodando em http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
