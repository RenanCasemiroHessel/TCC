"""
TCC FEI — Visualização das Imagens Segmentadas
Gera um mosaico mostrando: Original | Máscara | Imagem Segmentada (fundo preto)
para amostras de cada lote (B2, B3, B4)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import kagglehub

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
AMOSTRAS_POR_LOTE = 3   # quantas imagens mostrar por lote
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*60)
print("  TCC FEI — Visualização de Segmentação por Lote")
print("="*60 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# LOCALIZAR DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("[1/3] Localizando dataset via kagglehub ...")
dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

pastas_lote = {
    "B2 (Verde)":      (os.path.join(base_dir, "B2", "Side"), np.array([0, 255, 0, 255])),
    "B3 (Magenta)":    (os.path.join(base_dir, "B3"),          np.array([255, 30, 249, 255])),
    "B4 (Magenta)":    (os.path.join(base_dir, "B4", "Side"), np.array([255, 30, 249, 255])),
}

# ─────────────────────────────────────────────────────────────────────────────
# COLETAR AMOSTRAS
# ─────────────────────────────────────────────────────────────────────────────
print("[2/3] Coletando amostras de cada lote ...")

amostras = {}   # lote -> lista de (img_np, mask_np, seg_np, peso, img_file)

for lote_nome, (pasta, cor_alvo) in pastas_lote.items():
    pasta_imgs  = os.path.join(pasta, "images")
    pasta_masks = os.path.join(pasta, "annotations")
    amostras[lote_nome] = []

    if not os.path.exists(pasta_imgs):
        print(f"  [Aviso] Pasta não encontrada: {pasta_imgs}")
        continue

    count = 0
    for img_file in sorted(os.listdir(pasta_imgs)):
        if not img_file.endswith(".jpg"):
            continue

        img_path  = os.path.join(pasta_imgs, img_file)
        mask_path = os.path.join(pasta_masks, img_file + "___fuse.png")

        if not os.path.exists(mask_path):
            continue

        try:
            # Carregar imagem original
            img_np = np.array(Image.open(img_path).convert("RGB"))

            # Carregar máscara
            mask_pil = Image.open(mask_path)
            mask_np  = np.array(mask_pil)
            if mask_np.shape[-1] == 3:
                alpha   = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
                mask_np = np.concatenate([mask_np, alpha], axis=-1)

            # Criar segmentação (fundo preto)
            bovino_mask = np.all(mask_np == cor_alvo, axis=-1)
            seg_np      = img_np.copy()
            seg_np[~bovino_mask] = 0

            # Extrair peso do nome do arquivo
            lote_key = lote_nome.split(" ")[0]
            partes = img_file.replace(".jpg", "").split("_")
            try:
                indice = 3 if lote_key == "B4" else 2
                peso   = float(partes[indice])
            except Exception:
                peso = 0.0

            # Redimensionar para visualização (preservando aspecto)
            h, w = img_np.shape[:2]
            scale   = 400 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_small  = np.array(Image.fromarray(img_np).resize((new_w, new_h)))
            mask_vis   = mask_np[:, :, :3]   # só RGB da máscara
            mask_small = np.array(Image.fromarray(mask_vis).resize((new_w, new_h)))
            seg_small  = np.array(Image.fromarray(seg_np).resize((new_w, new_h)))

            amostras[lote_nome].append((img_small, mask_small, seg_small, peso, img_file))
            count += 1
            if count >= AMOSTRAS_POR_LOTE:
                break

        except Exception as e:
            print(f"  Erro ao processar {img_file}: {e}")
            continue

    print(f"  ✓ Lote {lote_nome}: {len(amostras[lote_nome])} amostras coletadas")

# ─────────────────────────────────────────────────────────────────────────────
# GERAR MOSAICO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Gerando mosaico de visualização ...")

n_lotes  = len(pastas_lote)
n_cols   = 3   # Original | Máscara | Segmentada

fig = plt.figure(figsize=(16, n_lotes * AMOSTRAS_POR_LOTE * 4 + 2))
fig.patch.set_facecolor("#1A1A2E")

titulo_geral = fig.suptitle(
    "TCC FEI — Visualização de Segmentação por Lote\n"
    "[ Original  |  Máscara de Cor  |  Imagem Segmentada (fundo preto) ]",
    fontsize=16, fontweight="bold", color="white", y=0.995
)

row_idx = 0
for lote_nome, lista in amostras.items():
    for amostra_idx, (img_s, mask_s, seg_s, peso, fname) in enumerate(lista):
        for col_idx, (titulo, img_data) in enumerate([
            ("ORIGINAL", img_s),
            ("MÁSCARA", mask_s),
            ("SEGMENTADA", seg_s),
        ]):
            ax_pos = row_idx * n_cols + col_idx + 1
            ax = fig.add_subplot(n_lotes * AMOSTRAS_POR_LOTE, n_cols, ax_pos)
            ax.imshow(img_data)
            ax.set_facecolor("#0D0D1A")

            if col_idx == 0:
                ax.set_ylabel(
                    f"{lote_nome}\n{peso:.0f} kg",
                    fontsize=9, color="white", rotation=0,
                    labelpad=55, va="center"
                )
            if amostra_idx == 0:
                ax.set_title(titulo, fontsize=10, fontweight="bold",
                             color="#00D4FF" if col_idx == 2 else "white")
            ax.axis("off")

        row_idx += 1

plt.tight_layout(rect=[0.08, 0, 1, 0.99])
out_path = os.path.join(SAVE_DIR, "visualizacao_segmentacao.png")
plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
plt.close()

print(f"\n  ✓ Mosaico salvo em: {out_path}")
print(f"\n{'='*60}")
print(f"  Abra o arquivo 'visualizacao_segmentacao.png' para ver")
print(f"  as imagens originais × máscaras × bois isolados!")
print(f"{'='*60}\n")
