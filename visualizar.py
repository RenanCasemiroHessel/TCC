import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

base = r".\data\www.acmeai.tech Dataset - BMGF-LivestockWeight-CV\Pixel"

COR_BOVINO = {
    "B2": np.array([0, 255, 0, 255]),
    "B3": np.array([255, 30, 249, 255]),
    "B4": np.array([255, 30, 249, 255]),
}

pastas_exemplo = {
    "B2": os.path.join(base, "B2", "Side"),
    "B3": os.path.join(base, "B3"),
    "B4": os.path.join(base, "B4", "Side"),
}

def extrair_peso(filename, lote):
    nome = filename.replace("___fuse.png", "").replace(".jpg", "")
    partes = nome.split("_")
    try:
        indice = 3 if lote == "B4" else 2
        val = float(partes[indice])
        if 50 <= val <= 1000:
            return val
    except:
        pass
    return None

def processar_exemplo(lote):
    pasta_imgs  = os.path.join(pastas_exemplo[lote], "images")
    pasta_masks = os.path.join(pastas_exemplo[lote], "annotations")

    if not os.path.exists(pasta_imgs):
        return None, None, None, None, None

    imagens = [f for f in os.listdir(pasta_imgs) if f.endswith(".jpg")]
    if not imagens:
        return None, None, None, None, None

    random.shuffle(imagens)
    for img_file in imagens:
        mask_file = img_file + "___fuse.png"
        mask_path = os.path.join(pasta_masks, mask_file)
        if not os.path.exists(mask_path):
            continue

        peso = extrair_peso(mask_file, lote)
        if peso is None:
            continue

        img  = np.array(Image.open(os.path.join(pasta_imgs, img_file)).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if mask.shape[-1] == 3:
            alpha = np.full((*mask.shape[:2], 1), 255, dtype=np.uint8)
            mask  = np.concatenate([mask, alpha], axis=-1)

        cor          = COR_BOVINO[lote]
        bovino_mask  = np.all(mask == cor, axis=-1)

        if bovino_mask.sum() == 0:
            continue

        overlay = img.copy()
        overlay[~bovino_mask] = (overlay[~bovino_mask] * 0.3).astype(np.uint8)
        overlay[bovino_mask]  = np.clip(
            overlay[bovino_mask] * 0.6 + np.array([0, 200, 0]) * 0.4, 0, 255
        ).astype(np.uint8)

        rows_idx = np.where(np.any(bovino_mask, axis=1))[0]
        cols_idx = np.where(np.any(bovino_mask, axis=0))[0]
        bbox     = (cols_idx[0], rows_idx[0], cols_idx[-1], rows_idx[-1])
        pixels   = int(bovino_mask.sum())

        return overlay, bovino_mask, bbox, peso, pixels

    return None, None, None, None, None


fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("Segmentação de Bovinos — Exemplos por Lote", fontsize=18, fontweight="bold", y=1.01)

for i, lote in enumerate(["B2", "B3", "B4"]):
    overlay, bovino_mask, bbox, peso, pixels = processar_exemplo(lote)

    ax_orig = axes[i][0]
    ax_seg  = axes[i][1]

    if overlay is None:
        ax_orig.set_title(f"Lote {lote} — não encontrado")
        ax_orig.axis("off")
        ax_seg.axis("off")
        continue

    pasta_imgs = os.path.join(pastas_exemplo[lote], "images")
    imagens    = [f for f in os.listdir(pasta_imgs) if f.endswith(".jpg")]
    img_orig = overlay.copy()  # fallback; overlay já tem a imagem base
    ax_orig.imshow(img_orig)
    ax_orig.set_title(f"Lote {lote} — Imagem com Segmentação\nPeso real: {peso} kg", fontsize=12)
    ax_orig.axis("off")

    x0, y0, x1, y1 = bbox
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2.5, edgecolor="lime", facecolor="none"
    )
    ax_orig.add_patch(rect)

    ax_seg.imshow(bovino_mask, cmap="Greens")
    ax_seg.set_title(
        f"Lote {lote} — Máscara do Bovino\n"
        f"Área: {pixels:,} px | BBox: {x1-x0}×{y1-y0} px",
        fontsize=12
    )
    ax_seg.axis("off")

plt.tight_layout()
plt.savefig("segmentacao_exemplo.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Imagem salva em: segmentacao_exemplo.png")


csv_path = "dataset_features_completo.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Distribuição do Dataset", fontsize=15, fontweight="bold")

    axes2[0].hist(df["peso_kg"], bins=30, color="#2ecc71", edgecolor="white")
    axes2[0].set_title("Distribuição de Pesos")
    axes2[0].set_xlabel("Peso (kg)")
    axes2[0].set_ylabel("Quantidade de animais")
    axes2[0].axvline(df["peso_kg"].mean(), color="red", linestyle="--", label=f"Média: {df['peso_kg'].mean():.0f} kg")
    axes2[0].legend()

    contagem = df["lote"].value_counts()
    axes2[1].bar(contagem.index, contagem.values, color=["#3498db", "#e74c3c", "#f39c12"], edgecolor="white")
    axes2[1].set_title("Amostras por Lote")
    axes2[1].set_xlabel("Lote")
    axes2[1].set_ylabel("Quantidade")
    for j, (idx, val) in enumerate(contagem.items()):
        axes2[1].text(j, val + 2, str(val), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("distribuicao_dataset.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Gráfico salvo em: distribuicao_dataset.png")
else:
    print("⚠️  dataset_features_completo.csv não encontrado. Rode o teste.py primeiro.")