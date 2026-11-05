"""
Preparação do Dataset Estendido (TCC2)
Escaneia o dataset completo do Kaggle (BMGF-LivestockWeight-CV) e monta os
pares side+rear por animal, SEM o corte em 250kg usado no TCC1 — mantém
qualquer peso plausível para bovino (30–1000 kg), removendo apenas valores
claramente resultantes de erro de parsing do nome do arquivo.

O lote B3 NÃO entra nos pares (não tem vista traseira, estrutura de pastas
diferente de B2/B4 — só images/ e annotations/ direto, sem Side/Rear). O
TCC1 o excluía inteiro rotulando como "dados clonados"; isso foi verificado
e não procede — comparação de hash/tamanho de arquivo e inspeção visual em
4 pares de mesmo peso confirmou que são animais diferentes, não duplicatas.
O B3 é salvo à parte, só como imagens laterais avulsas, para uso exclusivo
do treino single-view (não dá pra formar par sem a vista traseira).

Gera:
  dataset_pares_estendido.csv     (pares side+rear, B2+B4)
  dataset_singleview_b3.csv       (extra lateral-only, B3)
"""

import os
import numpy as np
import pandas as pd
import kagglehub

from caminhos import DADOS

DADOS_DIR = DADOS

WEIGHT_MIN_SANE = 30.0
WEIGHT_MAX_SANE = 1000.0   # descarta erros de parsing (ex.: 104343.0), não bovinos reais
LIMITE_TCC1 = 250.0        # só para o relatório comparativo abaixo

print("\n" + "=" * 60)
print("  Preparação do Dataset Estendido — TCC2")
print("=" * 60 + "\n")

print("Baixando/localizando dataset via kagglehub ...")
dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")


def parse_filename(filename, lote):
    name = filename.replace(".jpg", "")
    parts = name.split("_")
    try:
        if lote in ("B2", "B3"):   # B3 usa o mesmo formato de nome do B2 (id_vista_peso_sexo)
            animal_id = parts[0]
            weight = float(parts[2])
        else:  # B4
            animal_id = f"{parts[0]}_{parts[1]}"
            weight = float(parts[3])
        if WEIGHT_MIN_SANE <= weight <= WEIGHT_MAX_SANE:
            return animal_id, weight
    except Exception:
        pass
    return None


sides, rears = {}, {}
descartados_por_range = 0

for root, dirs, files in os.walk(base_dir):
    if os.path.basename(root) != "images":
        continue
    rel = os.path.relpath(root, base_dir)
    partes_path = rel.split(os.sep)
    lote = partes_path[0]
    if lote == "B3":  # dados clonados, mesmo critério do TCC1
        continue
    vista_dir = partes_path[1] if len(partes_path) > 1 else ""
    is_side = "Side" in vista_dir
    is_rear = "Back" in vista_dir or "Rear" in vista_dir
    if not (is_side or is_rear):
        continue

    pasta_masks = os.path.join(os.path.dirname(root), "annotations")
    if not os.path.exists(pasta_masks):
        continue

    for f in sorted(os.listdir(root)):
        if not f.endswith(".jpg"):
            continue
        res = parse_filename(f, lote)
        if res is None:
            descartados_por_range += 1
            continue
        animal_id, weight = res
        img_path = os.path.join(root, f)
        mask_path = os.path.join(pasta_masks, f + "___fuse.png")
        if not os.path.exists(mask_path):
            continue
        key = f"{lote}_{animal_id}_{weight}"
        entry = {"path": img_path, "mask": mask_path, "weight": weight, "lote": lote}
        if is_side and key not in sides:
            sides[key] = entry
        elif is_rear and key not in rears:
            rears[key] = entry

rows = []
for key in sides:
    if key in rears:
        rows.append({
            "key": key,
            "side_path": sides[key]["path"],
            "side_mask": sides[key]["mask"],
            "rear_path": rears[key]["path"],
            "rear_mask": rears[key]["mask"],
            "weight": sides[key]["weight"],
            "lote": sides[key]["lote"],
        })

df = pd.DataFrame(rows).drop_duplicates(subset="key").reset_index(drop=True)

out_path = os.path.join(DADOS_DIR, "dataset_pares_estendido.csv")
df.to_csv(out_path, index=False)

leves = df[df["weight"] < LIMITE_TCC1]
pesados = df[df["weight"] >= LIMITE_TCC1]

print(f"\nResultado:")
print(f"  Pares válidos (side+rear)       : {len(df)}")
print(f"  Faixa de peso                   : {df['weight'].min():.0f}–{df['weight'].max():.0f} kg")
print(f"  Registros descartados (fora do range plausível 30–1000kg) : {descartados_por_range}")
print(f"\n  Comparado ao corte do TCC1 (<{LIMITE_TCC1:.0f}kg):")
print(f"    Leves  (<{LIMITE_TCC1:.0f}kg)  : {len(leves):>4} pares  ({len(leves)/len(df)*100:.1f}%)")
print(f"    Pesados (>={LIMITE_TCC1:.0f}kg) : {len(pesados):>4} pares  ({len(pesados)/len(df)*100:.1f}%)  <- novo, ignorado no TCC1")

print(f"\n  Distribuição por faixa:")
faixas = [(30, 200, "Bezerro"), (200, 450, "Adulto"), (450, 1000, "Boi Gordo")]
for lo, hi, nome in faixas:
    n = len(df[(df["weight"] >= lo) & (df["weight"] < hi)])
    print(f"    {nome:10s} ({lo}-{hi}kg): {n:>4} pares")

print(f"\n  Salvo em: {out_path}")

# ── B3: extra lateral-only (sem par traseiro, estrutura de pasta própria) ──
print("\n" + "-" * 60)
print("  Lote B3 — extra single-view (lateral apenas)")
print("-" * 60)

b3_dir = os.path.join(base_dir, "B3")
b3_images = os.path.join(b3_dir, "images")
b3_masks = os.path.join(b3_dir, "annotations")

b3_rows = []
b3_descartados = 0
if os.path.isdir(b3_images) and os.path.isdir(b3_masks):
    for f in sorted(os.listdir(b3_images)):
        if not f.endswith(".jpg"):
            continue
        res = parse_filename(f, "B3")
        if res is None:
            b3_descartados += 1
            continue
        animal_id, weight = res
        img_path = os.path.join(b3_images, f)
        mask_path = os.path.join(b3_masks, f + "___fuse.png")
        if not os.path.exists(mask_path):
            continue
        b3_rows.append({
            "key": f"B3_{animal_id}_{weight}",
            "path": img_path,
            "mask": mask_path,
            "weight": weight,
            "lote": "B3",
            "vista": "side",
        })
else:
    print(f"  [!] Pasta do B3 não encontrada em: {b3_dir}")

df_b3 = pd.DataFrame(b3_rows).drop_duplicates(subset="key").reset_index(drop=True)
out_b3_path = os.path.join(DADOS_DIR, "dataset_singleview_b3.csv")
df_b3.to_csv(out_b3_path, index=False)

if len(df_b3) > 0:
    b3_pesados = df_b3[df_b3["weight"] >= LIMITE_TCC1]
    print(f"  Imagens válidas (B3, lateral)   : {len(df_b3)}")
    print(f"  Faixa de peso                   : {df_b3['weight'].min():.0f}–{df_b3['weight'].max():.0f} kg")
    print(f"  Descartadas (parsing/range)     : {b3_descartados}")
    print(f"  Pesados (>={LIMITE_TCC1:.0f}kg)              : {len(b3_pesados)}  ({len(b3_pesados)/len(df_b3)*100:.1f}%)")
    print(f"\n  Salvo em: {out_b3_path}")

print("=" * 60 + "\n")
