"""
CNN Single-View (ResNet18) calibrada pelo sticker de referência.

Testa se a mesma correção que quase dobrou o R² dos modelos tabulares
(comparar_sticker.py: LR R² 0,189 -> 0,406 só adicionando area_boi/area_sticker)
também ajuda a CNN, que em tese já "vê" a escala pelos pixels — mas sofre do
mesmo problema de fundo: duas fotos do mesmo boi, uma mais perto e outra mais
longe, viram áreas de pixel bem diferentes sem nenhuma pista de profundidade.

Duas correções aplicadas juntas (ver conversa — decisão de fazer as duas):

  1. Zoom de entrada: antes do resize final pra 224x224, cada imagem é
     redimensionada por um fator = diametro_sticker_referencia / diametro_sticker_nessa_foto,
     depois recortada numa janela fixa centrada no boi. Resultado: toda foto
     chega na rede com a mesma escala real (não só a área numérica como no
     tabular) — duas fotos do mesmo boi real, perto ou longe, ficam com o
     boi em tamanho aparente parecido no frame final.

  2. Feature auxiliar: o valor de sticker_ratio (mesmo usado no tabular)
     também é concatenado ao vetor de 512 features do ResNet antes da
     cabeça de regressão, do mesmo jeito que a arquitetura siamesa
     concatena side+rear. Deixa a rede aprender o peso ideal da pista,
     em vez de confiar só na correção geométrica do passo 1.

Entrada: dataset_features_sticker.csv (só lateral — sticker não aparece em
fotos traseiras, ver extrair_features_sticker.py).

Split único 80/20 por animal (não k-fold) — é o primeiro teste da hipótese,
k-fold fica pra depois se isso realmente ajudar.

ATENÇÃO — tempo de execução: ~mesma ordem de grandeza das outras CNNs do
projeto (minutos por época, dezenas de minutos a poucas horas pro treino
completo). Reduza NUM_EPOCHS pra uma validação rápida antes de rodar cheio.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_tcc import safe_path, salvar_json

# ── Configuração ──────────────────────────────────────────────────────────
SEED            = 42
IMG_SIZE        = 224
CANVAS_PX       = 2200     # janela de recorte (em pixels, após o zoom) centrada no boi
                            # medido nos dados: largura zoom p50=1540px, p95=1778px, p99=1945px,
                            # max=2197px — 2200 cobre praticamente todo mundo sem cortar
BATCH_SIZE      = 32
NUM_EPOCHS      = 30       # reduza pra ~8-10 numa validação rápida antes do treino "final"
FINETUNE_EPOCH  = 3
AUG_MULTIPLIER  = 4

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
MODELOS_DIR  = SAVE_DIR
os.makedirs(GRAFICOS_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# cores por lote, conforme Readme.md do dataset (mesmo usado em extrair_features_sticker.py)
CORES = {
    "B2": {"cattle": (0, 255, 0), "sticker": (255, 240, 0)},
    "B4": {"cattle": (255, 30, 249), "sticker": (0, 117, 255)},
}

print(f"  CNN Single-View calibrada pelo sticker — TCC2")
print(f"  Device      : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"  Épocas      : {NUM_EPOCHS}")

CSV_PATH = os.path.join(SAVE_DIR, "dataset_features_sticker.csv")
if not os.path.exists(CSV_PATH):
    print(f"\n  ERRO: {CSV_PATH} não encontrado.")
    print("  Rode primeiro: python extrair_features_sticker.py")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df = df[df["vista"] == "side"].reset_index(drop=True)  # sticker só existe em lateral

REF_STICKER_DIAM = float(np.sqrt(df["sticker_area"]).median())
print(f"  Amostras (lateral, com sticker): {len(df)}")
print(f"  Diâmetro de referência do sticker (mediana): {REF_STICKER_DIAM:.1f}px")
print(f"  Canvas de recorte pós-zoom      : {CANVAS_PX}px")

# normalização do escalar auxiliar (log + z-score, calculado sobre TODO o df
# só pra fins de escala — o fit real de treino/val seria idêntico já que é
# so log+padronização, sem vazamento de rótulo)
LOG_RATIO = np.log(df["sticker_ratio"].values)
RATIO_MEAN, RATIO_STD = float(LOG_RATIO.mean()), float(LOG_RATIO.std())


def normalizar_ratio(sticker_ratio):
    return (np.log(sticker_ratio) - RATIO_MEAN) / (RATIO_STD + 1e-8)


def carregar_escalado(img_path, mask_path, lote, sticker_area):
    """Segmenta, dá zoom proporcional ao sticker e recorta janela fixa centrada no boi."""
    try:
        img_np = np.array(Image.open(img_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path).convert("RGB"))
        cattle_rgb = np.array(CORES[lote]["cattle"])
        cattle_mask = np.all(mask_np == cattle_rgb, axis=-1)

        out = img_np.copy()
        out[~cattle_mask] = 0

        sticker_diam = float(np.sqrt(sticker_area))
        zoom = REF_STICKER_DIAM / sticker_diam

        h, w = out.shape[:2]
        new_h, new_w = max(8, int(h * zoom)), max(8, int(w * zoom))
        img_pil = Image.fromarray(out).resize((new_w, new_h), Image.LANCZOS)

        mask_pil = Image.fromarray((cattle_mask.astype(np.uint8)) * 255).resize(
            (new_w, new_h), Image.NEAREST
        )
        cattle_mask_zoomed = np.array(mask_pil) > 127
        rows = np.where(cattle_mask_zoomed.any(axis=1))[0]
        cols = np.where(cattle_mask_zoomed.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            cy, cx = new_h // 2, new_w // 2
        else:
            cy = int((rows.min() + rows.max()) / 2)
            cx = int((cols.min() + cols.max()) / 2)

        left, top = cx - CANVAS_PX // 2, cy - CANVAS_PX // 2
        cropped = img_pil.crop((left, top, left + CANVAS_PX, top + CANVAS_PX))
        return cropped
    except Exception:
        return Image.fromarray(np.zeros((CANVAS_PX, CANVAS_PX, 3), dtype=np.uint8))


class StickerScaledDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = carregar_escalado(row["path"], row["mask"], row["lote"], row["sticker_area"])
        if self.transform:
            img = self.transform(img)
        ratio = torch.tensor([normalizar_ratio(row["sticker_ratio"])], dtype=torch.float32)
        weight = torch.tensor(row["peso_kg"], dtype=torch.float32)
        return img, ratio, weight


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class StickerScaledNet(nn.Module):
    """ResNet18 + escalar do sticker concatenado antes da cabeça de regressão."""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512 + 1, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x, sticker_ratio):
        feat = self.backbone(x).flatten(1)
        feat = torch.cat([feat, sticker_ratio], dim=1)
        return self.head(feat)


# ── Split único 80/20 por animal ──────────────────────────────────────────
keys = df["id_animal"].unique()
train_keys, val_keys = train_test_split(keys, test_size=0.2, random_state=SEED)
train_df = df[df["id_animal"].isin(train_keys)].reset_index(drop=True)
val_df = df[df["id_animal"].isin(val_keys)].reset_index(drop=True)

print(f"  Treino: {len(train_df)} imgs ({len(train_keys)} animais) | "
      f"Val: {len(val_df)} imgs ({len(val_keys)} animais)")

Y_MIN = float(train_df["peso_kg"].min())
Y_MAX = float(train_df["peso_kg"].max())

train_dataset = ConcatDataset([StickerScaledDataset(train_df, train_transform)] * AUG_MULTIPLIER)
val_dataset = StickerScaledDataset(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=torch.cuda.is_available())

model = StickerScaledNet().to(DEVICE)
for name, param in model.named_parameters():
    if "backbone" in name:
        param.requires_grad = False

huber_delta = 50.0 / (Y_MAX - Y_MIN)
criterion = nn.HuberLoss(delta=huber_delta)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
best_epoch = 0
model_path = safe_path(os.path.join(MODELOS_DIR, "resnet18_singleview_sticker.pth"))

print(f"\n  Treinando -> {model_path}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    if epoch == FINETUNE_EPOCH:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    model.train()
    running = 0.0
    for imgs, ratios, weights in train_loader:
        imgs, ratios = imgs.to(DEVICE), ratios.to(DEVICE)
        w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs, ratios)
        loss = criterion(preds, w_norm)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    train_loss = running / len(train_dataset)

    model.eval()
    val_run = 0.0
    with torch.no_grad():
        for imgs, ratios, weights in val_loader:
            imgs, ratios = imgs.to(DEVICE), ratios.to(DEVICE)
            w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
            preds = model(imgs, ratios)
            val_run += criterion(preds, w_norm).item() * imgs.size(0)
    val_loss = val_run / len(val_dataset)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    marker = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({"model_state_dict": model.state_dict(),
                    "y_min": Y_MIN, "y_max": Y_MAX, "img_size": IMG_SIZE,
                    "ref_sticker_diam": REF_STICKER_DIAM, "canvas_px": CANVAS_PX,
                    "ratio_mean": RATIO_MEAN, "ratio_std": RATIO_STD}, model_path)
        marker = "  * melhor"

    print(f"    Época {epoch:02d}/{NUM_EPOCHS} | Train: {train_loss:.6f} | "
          f"Val: {val_loss:.6f} | {time.time()-t0:.1f}s{marker}")

# ── Avaliação final com o melhor checkpoint ────────────────────────────────
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

preds_kg, true_kg = [], []
with torch.no_grad():
    for imgs, ratios, weights in val_loader:
        imgs, ratios = imgs.to(DEVICE), ratios.to(DEVICE)
        preds = model(imgs, ratios).squeeze(1).cpu().numpy()
        preds_kg.extend((preds * (Y_MAX - Y_MIN) + Y_MIN).tolist())
        true_kg.extend(weights.numpy().tolist())

preds_kg, true_kg = np.array(preds_kg), np.array(true_kg)
mae = mean_absolute_error(true_kg, preds_kg)
rmse = np.sqrt(mean_squared_error(true_kg, preds_kg))
r2 = r2_score(true_kg, preds_kg)
pearson = float(np.corrcoef(true_kg, preds_kg)[0, 1])
acc10 = float((np.abs(preds_kg - true_kg) / true_kg <= 0.10).mean() * 100)

print(f"""
{'='*60}
  RESULTADO — CNN Single-View calibrada pelo sticker
{'='*60}

  Melhor época : {best_epoch}/{NUM_EPOCHS}
  MAE          : {mae:.2f} kg
  RMSE         : {rmse:.2f} kg
  R²           : {r2:.4f}
  Pearson r    : {pearson:.4f}
  Acc ±10%     : {acc10:.1f}%

  Comparar com:
    CNN Single-View original (TCC1, dataset maior side+rear): MAE 20,95kg | R² 0,3718
    LR tabular + sticker_ratio (mesmo dataset, 2401 lateral) : MAE 21,03kg | R² 0,4058
""")

fig, ax = plt.subplots(figsize=(9, 5))
ep = range(1, NUM_EPOCHS + 1)
ax.plot(ep, history["train_loss"], label="Treino")
ax.plot(ep, history["val_loss"], label="Validação")
ax.set_title("CNN Single-View (sticker) — Curva de Loss", fontsize=14, fontweight="bold")
ax.set_xlabel("Época"); ax.set_ylabel("Huber Loss (normalizado)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "singleview_sticker_curva_loss.png"), dpi=150)
plt.close(fig)

np.save(safe_path(os.path.join(SAVE_DIR, "predictions_cnn_singleview_sticker.npy")),
        np.array([true_kg, preds_kg]))

salvar_json(os.path.join(SAVE_DIR, "resultados_cnn_singleview_sticker.json"),
            "CNN Single-View + sticker (zoom + escalar)", {
                "num_epochs": NUM_EPOCHS, "best_epoch": best_epoch,
                "n_treino": len(train_df), "n_val": len(val_df),
                "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
                "pearson_r": pearson, "acc10": acc10,
                "ref_sticker_diam_px": REF_STICKER_DIAM, "canvas_px": CANVAS_PX,
            })

print(f"  Modelo salvo em : {model_path}")
print(f"  Gráfico salvo em: {os.path.join(GRAFICOS_DIR, 'singleview_sticker_curva_loss.png')}")
