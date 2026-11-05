"""
Extração de features morfométricas + feature calibrada pelo sticker de referência.

O script que originalmente gerou dataset_features_completo.csv não está no repo,
então este aqui reconstrói a extração do zero a partir das máscaras brutas do
Kaggle (lotes B2 + B4, side + rear), reproduzindo as mesmas definições de
area_norm (= area / (altura_img * largura_img)) e proporcao (= largura/altura)
que batem com os valores já salvos em dataset_features_completo.csv.

Novidade: o Readme oficial do dataset documenta uma classe de segmentação
"Sticker" — um adesivo de referência colado no boi para corrigir variação de
distância da câmera entre fotos (ver PDF do dataset). O pipeline atual ignora
essa classe. Aqui ela é extraída e usada para normalizar a área/dimensões do
boi pela área do sticker, removendo parte do efeito "foto tirada mais perto
ou mais longe" que contamina a área bruta em pixels.

Saída: dataset_features_sticker.csv (lotes B2+B4, side+rear, 50<=peso<250kg)
"""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import kagglehub

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_MIN, WEIGHT_MAX = 50.0, 250.0

# cores por lote, conforme Readme.md do dataset (RGB)
CORES = {
    "B2": {"cattle": (0, 255, 0), "sticker": (255, 240, 0)},
    "B4": {"cattle": (255, 30, 249), "sticker": (0, 117, 255)},
}


def parse_filename(filename, lote):
    name = filename.replace(".jpg", "")
    parts = name.split("_")
    try:
        if lote == "B2":
            animal_id = parts[0]
            weight = float(parts[2])
            sexo = parts[-1]
        else:  # B4
            animal_id = f"{parts[0]}_{parts[1]}"
            weight = float(parts[3])
            sexo = parts[-1]
        if WEIGHT_MIN <= weight < WEIGHT_MAX and sexo in ("M", "F"):
            return animal_id, weight, sexo
    except Exception:
        pass
    return None


def extrair_linha(img_path, mask_path, lote, animal_id, weight, sexo, vista):
    mask_np = np.array(Image.open(mask_path).convert("RGB"))
    h, w = mask_np.shape[:2]

    cattle_rgb = np.array(CORES[lote]["cattle"])
    sticker_rgb = np.array(CORES[lote]["sticker"])

    cattle_mask = np.all(mask_np == cattle_rgb, axis=-1)
    sticker_mask = np.all(mask_np == sticker_rgb, axis=-1)

    area = int(cattle_mask.sum())
    sticker_area = int(sticker_mask.sum())
    if area < 1000 or sticker_area < 50:
        return None  # segmentação falhou ou sticker não visível/oclusão

    rows = np.where(cattle_mask.any(axis=1))[0]
    cols = np.where(cattle_mask.any(axis=0))[0]
    altura_bbox = int(rows.max() - rows.min() + 1)
    largura_bbox = int(cols.max() - cols.min() + 1)

    area_norm = area / (h * w)
    proporcao = largura_bbox / altura_bbox

    sticker_diam_px = float(np.sqrt(sticker_area))  # proxy do diâmetro do marcador em pixels
    sticker_ratio = area / sticker_area              # área do boi em "unidades de sticker"
    altura_cal = altura_bbox / sticker_diam_px
    largura_cal = largura_bbox / sticker_diam_px

    return {
        "area": area,
        "altura_bbox": altura_bbox,
        "largura_bbox": largura_bbox,
        "proporcao": proporcao,
        "area_norm": area_norm,
        "sticker_area": sticker_area,
        "sticker_ratio": sticker_ratio,
        "altura_cal": altura_cal,
        "largura_cal": largura_cal,
        "id_animal": f"{lote}_{animal_id}",
        "sexo": sexo,
        "vista": vista,
        "peso_kg": weight,
        "lote": lote,
        "path": img_path,
        "mask": mask_path,
    }


def main():
    print("\n" + "=" * 60)
    print("  Extração de features + calibração por sticker")
    print("=" * 60 + "\n")

    dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
    base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

    rows = []
    descartadas_peso, descartadas_seg = 0, 0

    for root, _dirs, files in os.walk(base_dir):
        if os.path.basename(root) != "images":
            continue
        rel = os.path.relpath(root, base_dir)
        partes_path = rel.split(os.sep)
        lote = partes_path[0]
        if lote not in CORES:  # só B2 e B4 (B3 fica de fora, igual ao TCC1)
            continue
        vista_dir = partes_path[1] if len(partes_path) > 1 else ""
        is_side = "Side" in vista_dir
        is_rear = "Back" in vista_dir or "Rear" in vista_dir
        if not (is_side or is_rear):
            continue
        vista = "side" if is_side else "rear"

        pasta_masks = os.path.join(os.path.dirname(root), "annotations")
        if not os.path.exists(pasta_masks):
            continue

        for f in sorted(os.listdir(root)):
            if not f.endswith(".jpg"):
                continue
            res = parse_filename(f, lote)
            if res is None:
                descartadas_peso += 1
                continue
            animal_id, weight, sexo = res
            img_path = os.path.join(root, f)
            mask_path = os.path.join(pasta_masks, f + "___fuse.png")
            if not os.path.exists(mask_path):
                continue
            linha = extrair_linha(img_path, mask_path, lote, animal_id, weight, sexo, vista)
            if linha is None:
                descartadas_seg += 1
                continue
            rows.append(linha)

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["id_animal", "vista", "peso_kg"]
    ).reset_index(drop=True)

    out_path = os.path.join(SAVE_DIR, "dataset_features_sticker.csv")
    df.to_csv(out_path, index=False)

    print(f"  Imagens válidas             : {len(df)}")
    print(f"  Descartadas (peso/sexo)     : {descartadas_peso}")
    print(f"  Descartadas (sem sticker/seg): {descartadas_seg}")
    print(f"  Lotes                        : {df['lote'].value_counts().to_dict()}")
    print(f"  Vista                        : {df['vista'].value_counts().to_dict()}")
    print(f"\n  Salvo em: {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
