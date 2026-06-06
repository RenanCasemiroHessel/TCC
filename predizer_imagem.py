"""
Inferência por Imagem Única (com segmentação YOLO automática)
Carrega o modelo ResNet18 treinado e prevê o peso de um bovino a partir de uma imagem.
A segmentação é feita automaticamente via YOLOv11-seg (detecta a classe "cow" do COCO).

Uso:
    python predizer_imagem.py
    python predizer_imagem.py --imagem foto.jpg
    python predizer_imagem.py --imagem foto.jpg --modelo outro_modelo.pth
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FAIXAS = [
    (0,   250, "Leve    (<250 kg)"),
    (250, 450, "Médio   (250–450 kg)"),
    (450, 9999,"Pesado  (>450 kg)"),
]

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELO_PADRAO = os.path.join(SCRIPT_DIR, "resnet18_gado_segmentado.pth")
YOLO_PATH     = os.path.join(SCRIPT_DIR, "yolo11n-seg.pt")

COCO_COW_CLASS = 19


def carregar_modelo(caminho_pth: str, device: torch.device):
    if not os.path.exists(caminho_pth):
        print(f"\n[ERRO] Arquivo de modelo não encontrado: {caminho_pth}")
        print("  Execute primeiro: python treinar_cnn.py")
        sys.exit(1)

    checkpoint = torch.load(caminho_pth, map_location=device, weights_only=False)

    img_size = checkpoint.get("img_size", 224)
    y_min    = checkpoint["y_min"]
    y_max    = checkpoint["y_max"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, y_min, y_max, img_size

def aplicar_yolo_mascara(img_path: str) -> Image.Image:
    """Segmenta o boi via YOLO e retorna imagem com fundo preto."""
    from ultralytics import YOLO

    if not os.path.exists(YOLO_PATH):
        print(f"  [YOLO] Modelo não encontrado em {YOLO_PATH}.")
        print("  Instale com: pip install ultralytics")
        print("  O modelo yolo11n-seg.pt será baixado automaticamente na primeira execução.")

    yolo = YOLO(YOLO_PATH)

    img_pil = Image.open(img_path).convert("RGB")
    img_np  = np.array(img_pil)
    h, w    = img_np.shape[:2]

    results = yolo(img_path, conf=0.3, classes=[COCO_COW_CLASS], verbose=False)

    if results and results[0].masks is not None and len(results[0].masks) > 0:
        combined_mask = np.zeros((h, w), dtype=bool)
        for mask_tensor in results[0].masks.data:
            mask_bin = mask_tensor.cpu().numpy()
            mask_resized = np.array(
                Image.fromarray((mask_bin * 255).astype(np.uint8)).resize((w, h))
            ) > 127
            combined_mask |= mask_resized

        masked_img = img_np.copy()
        masked_img[~combined_mask] = 0
        print(f"  [YOLO] Boi detectado e segmentado com sucesso! "
              f"({combined_mask.sum()} pixels de boi)")
        return Image.fromarray(masked_img)

    dirname, filename = os.path.split(img_path)
    parent_dir, last_dir = os.path.split(dirname)
    if last_dir == "images":
        mask_path = os.path.join(parent_dir, "annotations", filename + "___fuse.png")
        if os.path.exists(mask_path):
            mask_pil = Image.open(mask_path)
            mask_np  = np.array(mask_pil)
            if mask_np.shape[-1] == 3:
                alpha   = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
                mask_np = np.concatenate([mask_np, alpha], axis=-1)
            verde   = np.array([0, 255, 0, 255])
            magenta = np.array([255, 30, 249, 255])
            for cor_alvo in [verde, magenta]:
                bovino_mask = np.all(mask_np == cor_alvo, axis=-1)
                if bovino_mask.sum() > 100:
                    masked_img = img_np.copy()
                    masked_img[~bovino_mask] = 0
                    print("  [Fallback] Máscara colorida do dataset encontrada e aplicada.")
                    return Image.fromarray(masked_img)

    print("  [AVISO] Nenhuma segmentação encontrada. Usando imagem original.")
    return img_pil

def classificar_faixa(peso_kg: float) -> str:
    for baixo, alto, nome in FAIXAS:
        if baixo <= peso_kg < alto:
            return nome
    return "Indefinida"

def predizer(img_path: str, modelo_pth: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Inferência de Peso Bovino")
    print(f"{'='*55}")
    print(f"  Dispositivo : {device}")
    print(f"  Modelo      : {os.path.basename(modelo_pth)}")
    print(f"  Imagem      : {os.path.basename(img_path)}")
    print(f"{'='*55}\n")

    print("  Carregando modelo ...")
    model, y_min, y_max, img_size = carregar_modelo(modelo_pth, device)

    print("  Segmentando boi (YOLO automático) ...")
    if not os.path.exists(img_path):
        print(f"\n[ERRO] Imagem não encontrada: {img_path}")
        sys.exit(1)

    img_segmentada = aplicar_yolo_mascara(img_path)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(img_segmentada).unsqueeze(0).to(device)

    print("  Realizando predição ...")
    with torch.no_grad():
        saida_norm = model(tensor).squeeze().item()

    peso_kg = saida_norm * (y_max - y_min) + y_min
    faixa   = classificar_faixa(peso_kg)

    print(f"\n{'='*55}")
    print(f"  RESULTADO")
    print(f"{'='*55}")
    print(f"  Peso estimado : {peso_kg:.1f} kg")
    print(f"  Faixa         : {faixa}")
    print(f"{'='*55}\n")

    return peso_kg


def pedir_imagem() -> str:
    while True:
        caminho = input("  Caminho da imagem (ex: /home/user/foto.jpg): ").strip()
        if os.path.exists(caminho):
            return caminho
        print(f"  Arquivo não encontrado: {caminho}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predição de peso bovino por imagem (YOLO automático)")
    parser.add_argument("--imagem",  type=str, default=None,        help="Caminho para a imagem")
    parser.add_argument("--modelo",  type=str, default=MODELO_PADRAO, help="Caminho para o .pth")
    args = parser.parse_args()

    img_path = args.imagem if args.imagem else pedir_imagem()

    predizer(img_path, args.modelo)
