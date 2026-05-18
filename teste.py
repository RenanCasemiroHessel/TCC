import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

base = r".\data\www.acmeai.tech Dataset - BMGF-LivestockWeight-CV\Pixel"

# Cor do bovino por lote (conforme visualizar.py)
COR_BOVINO = {
    "B2": np.array([0, 255, 0, 255]),       # verde
    "B3": np.array([255, 30, 249, 255]),    # rosa
    "B4": np.array([255, 30, 249, 255]),    # rosa (igual B3)
}

def extrair_metadados(filename, lote, pasta_caminho):
    nome = filename.replace("___fuse.png", "").replace(".jpg", "")
    partes = nome.split("_")
    try:
        # B2: Ex: 1.0_s_117_10.0_F -> partes: ['1.0', 's', '117', '10.0', 'F']
        # B3: Ex: 100_s_109_M -> partes: ['100', 's', '109', 'M']
        # B4: Ex: 100_b4-1_s_124_F -> partes: ['100', 'b4-1', 's', '124', 'F']
        
        if lote == "B2":
            id_animal = f"B2_{partes[0]}"
            vista = partes[1]
            peso = float(partes[2])
            sexo = partes[4]
        elif lote == "B3":
            id_animal = f"B3_{partes[0]}"
            vista = partes[1]
            peso = float(partes[2])
            sexo = partes[3]
        elif lote == "B4":
            id_animal = f"B4_{partes[0]}_{partes[1]}" # ID unico contendo o lote/sublote
            vista = partes[2]
            peso = float(partes[3])
            sexo = partes[4]
        else:
            return None
            
        # Normalização do Sexo (M = Macho, F = Fêmea)
        sexo = sexo.upper().strip()
        if "M" in sexo:
            sexo = "M"
        elif "F" in sexo:
            sexo = "F"
        else:
            sexo = "N/A"
            
        # Normalização da Vista (Side/Lateral vs Back/Rear)
        vista = vista.lower().strip()
        if vista == "s":
            vista = "side"
        elif vista in ["r", "b"]:
            vista = "back"
        else:
            # Fallback para o nome do caminho da pasta
            pasta_lower = pasta_caminho.lower()
            if "side" in pasta_lower:
                vista = "side"
            elif "back" in pasta_lower or "rear" in pasta_lower:
                vista = "back"
            else:
                vista = "side" # B3 padrão
                
        if 50 <= peso <= 1000:
            return {
                "id_animal": id_animal,
                "sexo": sexo,
                "vista": vista,
                "peso_kg": peso
            }
    except Exception as e:
        pass
    return None

def extrair_features(mask_path, cor_bovino):
    mask = np.array(Image.open(mask_path))

    # Garante canal alpha (RGBA)
    if mask.shape[-1] == 3:
        alpha = np.full((*mask.shape[:2], 1), 255, dtype=np.uint8)
        mask = np.concatenate([mask, alpha], axis=-1)

    bovino = np.all(mask == cor_bovino, axis=-1)
    if bovino.sum() == 0:
        return None

    rows_idx = np.where(np.any(bovino, axis=1))[0]
    cols_idx = np.where(np.any(bovino, axis=0))[0]

    altura  = int(rows_idx[-1] - rows_idx[0])
    largura = int(cols_idx[-1] - cols_idx[0])
    area    = int(bovino.sum())
    proporcao = round(largura / altura, 4) if altura > 0 else 0

    return {
        "area":         area,
        "altura_bbox":  altura,
        "largura_bbox": largura,
        "proporcao":    proporcao,
        "area_norm":    round(area / (mask.shape[0] * mask.shape[1]), 6)
    }

# Define quais pastas processar por lote
fontes = [
    ("B2", os.path.join(base, "B2", "Side",   "annotations")),
    ("B2", os.path.join(base, "B2", "Side_2", "annotations")),
    ("B2", os.path.join(base, "B2", "Back",   "annotations")),
    ("B2", os.path.join(base, "B2", "Back_2", "annotations")),
    ("B3", os.path.join(base, "B3",           "annotations")),
    ("B4", os.path.join(base, "B4", "Side",   "annotations")),
    ("B4", os.path.join(base, "B4", "Rear",   "annotations")),
]

registros = []
ignorados = 0

for lote, pasta in fontes:
    if not os.path.exists(pasta):
        print(f"[PASTA NÃO ENCONTRADA] {pasta}")
        continue

    arquivos = sorted(os.listdir(pasta))
    print(f"\nProcessando {lote} — {os.path.basename(os.path.dirname(pasta))}: {len(arquivos)} máscaras")

    for mask_file in tqdm(arquivos, desc=f"{lote}"):
        meta = extrair_metadados(mask_file, lote, pasta)
        if meta is None:
            ignorados += 1
            continue

        features = extrair_features(os.path.join(pasta, mask_file), COR_BOVINO[lote])
        if features is None:
            ignorados += 1
            continue

        # Une as características geométricas aos metadados (id_animal, sexo, vista, peso)
        features.update(meta)
        features["lote"] = lote
        registros.append(features)

df = pd.DataFrame(registros)
df.to_csv("dataset_features_completo.csv", index=False)

print(f"\n{'='*40}")
print(f"Total amostras válidas : {len(df)}")
print(f"Total ignoradas        : {ignorados}")
print(f"\nAmostras por lote:")
print(df["lote"].value_counts())
print(f"\nDistribuição por Sexo:")
print(df["sexo"].value_counts())
print(f"\nDistribuição por Vista:")
print(df["vista"].value_counts())
print(f"\nEstatísticas de peso:")
print(df["peso_kg"].describe())