"""
CNN Multi-View (ResNet18 Siamesa) para Estimativa de Peso Bovino
Usa par de imagens (lateral + traseira) do mesmo animal para predição.
Split por animal (sem data leakage).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,"figure.titlesize":17})
import time
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import kagglehub

# CONFIG
SEED           = 42
IMG_SIZE       = 224
BATCH_SIZE     = 16      # 2 imagens por amostra → metade do batch single-view
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-4
WEIGHT_MAX     = 250.0
SAMPLE_FRAC    = 1.0     # 1.0 = dataset completo; 0.1 = teste rápido
FINETUNE_EPOCH = 3       # descongelar backbone a partir desta época
AUG_MULTIPLIER = 4       # repete dataset de treino N vezes com augmentações diferentes

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

print(f"  CNN Multi-View (ResNet18 Siamesa)")
print(f"  Device     : {DEVICE}")
print(f"  Épocas     : {NUM_EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  IMG_SIZE   : {IMG_SIZE}x{IMG_SIZE}")

# 1. SCAN DO DATASET — coleta pares (lateral, traseira) por animal
print("Escaneando dataset e montando pares side+rear ...")

dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

def parse_filename(filename, lote):
    name = filename.replace(".jpg", "")
    parts = name.split("_")
    try:
        if lote == "B2":
            animal_id = parts[0]
            weight    = float(parts[2])
        else:  # B4
            animal_id = f"{parts[0]}_{parts[1]}"
            weight    = float(parts[3])
        if 50 <= weight < WEIGHT_MAX:
            return animal_id, weight
    except Exception:
        pass
    return None

def cor_mascara(img_path):
    return np.array([0, 255, 0, 255]) if "B2" in img_path else np.array([255, 30, 249, 255])

sides = {}  # key → {"path", "mask", "weight", "lote"}
rears = {}

for root, dirs, files in os.walk(base_dir):
    if os.path.basename(root) != "images":
        continue
    rel          = os.path.relpath(root, base_dir)
    partes_path  = rel.split(os.sep)
    lote         = partes_path[0]
    if lote == "B3":
        continue
    vista_dir    = partes_path[1] if len(partes_path) > 1 else ""
    is_side      = "Side" in vista_dir
    is_rear      = "Back" in vista_dir or "Rear" in vista_dir
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
            continue
        animal_id, weight = res
        img_path  = os.path.join(root, f)
        mask_path = os.path.join(pasta_masks, f + "___fuse.png")
        if not os.path.exists(mask_path):
            continue
        key = f"{lote}_{animal_id}_{weight}"
        entry = {"path": img_path, "mask": mask_path, "weight": weight, "lote": lote}
        if is_side and key not in sides:
            sides[key] = entry
        elif is_rear and key not in rears:
            rears[key] = entry

# Montar DataFrame de pares
rows = []
for key in sides:
    if key in rears:
        rows.append({
            "key":        key,
            "side_path":  sides[key]["path"],
            "side_mask":  sides[key]["mask"],
            "rear_path":  rears[key]["path"],
            "rear_mask":  rears[key]["mask"],
            "weight":     sides[key]["weight"],
            "lote":       sides[key]["lote"],
        })

df = pd.DataFrame(rows)

# ── Log detalhado do dataset ──────────────────────────────────────────────────
total_imagens = len(df) * 2  # side + rear por par
lotes_count   = df["lote"].value_counts().to_dict()
print(f"""
    print("\nResultados:")
    print(f"Pares válidos (side+rear)  : {len(df):>6}")
    print(f"Imagens únicas observadas  : {total_imagens:>6} (2 por par)")
    print(f"Lote B2  : {lotes_count.get('B2',0):>6} pares")
    print(f"Lote B4  : {lotes_count.get('B4',0):>6} pares")
    print(f"Filtro de peso  : 50–{WEIGHT_MAX:.0f} kg")
    print(f"ConcatDataset {AUG_MULTIPLIER}x → {len(df)*AUG_MULTIPLIER:>6} pares efetivos/época")
""")

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)

# 2. SPLIT 80/20 POR ANIMAL (sem data leakage)
print("\nDividindo dados por animal (80/20) ...")

keys = df["key"].tolist()
train_keys, val_keys = train_test_split(keys, test_size=0.2, random_state=SEED)

train_df = df[df["key"].isin(train_keys)].reset_index(drop=True)
val_df   = df[df["key"].isin(val_keys)].reset_index(drop=True)

# 3. DATASET E TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_segmented(img_path, mask_path):
    """Carrega imagem com fundo removido usando máscara anotada."""
    try:
        img_np  = np.array(Image.open(img_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path))
        if mask_np.shape[-1] == 3:
            alpha   = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
            mask_np = np.concatenate([mask_np, alpha], axis=-1)
        cor     = cor_mascara(img_path)
        mask    = np.all(mask_np == cor, axis=-1)
        out     = img_np.copy()
        out[~mask] = 0
        return Image.fromarray(out)
    except Exception:
        return Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

class MultiViewDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        side_img = load_segmented(row["side_path"], row["side_mask"])
        rear_img = load_segmented(row["rear_path"], row["rear_mask"])
        weight   = torch.tensor(row["weight"], dtype=torch.float32)
        if self.transform:
            side_img = self.transform(side_img)
            rear_img = self.transform(rear_img)
        return side_img, rear_img, weight

train_dataset_base = MultiViewDataset(train_df, train_transform)
val_dataset        = MultiViewDataset(val_df,   val_transform)

# ConcatDataset: cada cópia vê augmentações aleatórias diferentes → sem custo em disco
from torch.utils.data import ConcatDataset
train_dataset = ConcatDataset([train_dataset_base] * AUG_MULTIPLIER)

print(f"      Augmentação {AUG_MULTIPLIER}x: {len(train_dataset_base)} → {len(train_dataset)} pares efetivos de treino")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=torch.cuda.is_available())

# 4. MODELO — ResNet18 Siamesa (backbone compartilhado)
print("\nConstruindo modelo (ResNet18 Siamesa) ...")

train_weights = train_df["weight"].values
Y_MIN = float(train_weights.min())
Y_MAX = float(train_weights.max())

class SiameseResNet(nn.Module):
    """
    Backbone ResNet18 compartilhado entre vista lateral e traseira.
    Features de ambas as vistas são concatenadas antes da cabeça de regressão.
    """
    def __init__(self):
        super().__init__()
        base          = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)
        self.head     = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_side, x_rear):
        feat_side = self.backbone(x_side).flatten(1)           # (B, 512)
        feat_rear = self.backbone(x_rear).flatten(1)           # (B, 512)
        feat      = torch.cat([feat_side, feat_rear], dim=1)   # (B, 1024)
        return self.head(feat)

model = SiameseResNet().to(DEVICE)

# Fase 1: congelar backbone
for name, param in model.named_parameters():
    if "backbone" in name:
        param.requires_grad = False

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

huber_delta = 50.0 / (Y_MAX - Y_MIN)
criterion   = nn.HuberLoss(delta=huber_delta)
optimizer   = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=LEARNING_RATE, weight_decay=1e-4)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

# 5. LOOP DE TREINO
print(f"\nIniciando treino por {NUM_EPOCHS} épocas ...")
print(f"      Épocas 1-{FINETUNE_EPOCH-1}: só cabeça | Época {FINETUNE_EPOCH}+: backbone descongelado\n")

history       = {"train_loss": [], "val_loss": []}
from utils_tcc import safe_path, salvar_json
MODEL_PATH = safe_path(os.path.join(SAVE_DIR, "resnet18_multiview_gado.pth"))
print(f"      Modelo será salvo em: {os.path.basename(MODEL_PATH)}")

best_val_loss = float("inf")
best_epoch    = 0

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    if epoch == FINETUNE_EPOCH:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)
        print(f"  [Época {epoch}] Backbone DESCONGELADO — LR={LEARNING_RATE * 0.1:.6f}\n")

    # ── Treino ──
    model.train()
    running = 0.0
    for side, rear, weights in train_loader:
        side, rear = side.to(DEVICE), rear.to(DEVICE)
        w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        preds  = model(side, rear)
        loss   = criterion(preds, w_norm)
        loss.backward()
        optimizer.step()
        running += loss.item() * side.size(0)
    train_loss = running / len(train_dataset)

    # ── Validação ──
    model.eval()
    val_run = 0.0
    with torch.no_grad():
        for side, rear, weights in val_loader:
            side, rear = side.to(DEVICE), rear.to(DEVICE)
            w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
            preds  = model(side, rear)
            val_run += criterion(preds, w_norm).item() * side.size(0)
    val_loss = val_run / len(val_dataset)

    scheduler.step()
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    marker = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "y_min": Y_MIN, "y_max": Y_MAX, "img_size": IMG_SIZE,
        }, MODEL_PATH)
        marker = "  * melhor val loss"

    elapsed = time.time() - t0
    print(f"  Época {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"Tempo: {elapsed:.1f}s{marker}")

print(f"\n  Melhor modelo: época {best_epoch} | Val Loss: {best_val_loss:.6f}")

# 6. AVALIAÇÃO FINAL (usando melhor checkpoint)
print("\nCalculando métricas finais ...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"      Usando pesos da época {best_epoch} (val loss: {best_val_loss:.6f})")

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for side, rear, weights in val_loader:
        side, rear = side.to(DEVICE), rear.to(DEVICE)
        preds      = model(side, rear).squeeze(1).cpu().numpy()
        preds_kg   = preds * (Y_MAX - Y_MIN) + Y_MIN
        all_preds.extend(preds_kg.tolist())
        all_true.extend(weights.numpy().tolist())

y_true = np.array(all_true)
y_pred = np.array(all_preds)
errors = y_pred - y_true

mae       = mean_absolute_error(y_true, y_pred)
rmse      = np.sqrt(mean_squared_error(y_true, y_pred))
r2        = r2_score(y_true, y_pred)
pearson_r = np.corrcoef(y_true, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — CNN Multi-View (ResNet18 Siamesa)
{'='*60}

    print("\nResultados:")
    print(f"MAE  (Erro Absoluto Médio)  : {mae:>8.2f} kg")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg")
    print(f"R²  (Coef. de Determinação)  : {r2:>8.4f}")
    print(f"Pearson r  : {pearson_r:>8.4f}")
""")

# 7. GRÁFICOS
print("  Gerando gráficos ...\n")
DPI = 150

# 7.1 Curva de Loss
fig, ax = plt.subplots(figsize=(9, 5))
ep = range(1, NUM_EPOCHS + 1)
ax.plot(ep, history["train_loss"], "o-", color="#2563EB", linewidth=2.5, markersize=6, label="Treino")
ax.plot(ep, history["val_loss"],   "s--", color="#DC2626", linewidth=2.5, markersize=6, label="Validação")
ax.axvline(best_epoch, color="#F59E0B", linestyle=":", linewidth=1.5, label=f"Melhor época ({best_epoch})")
ax.set_title("CNN Multi-View — Curva de Loss", fontsize=14, fontweight="bold")
ax.set_xlabel("Época"); ax.set_ylabel("Huber Loss (normalizado)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "cnn_multiview_curva_loss.png"), dpi=DPI)
plt.close(fig)
print("  [OK] cnn_multiview_curva_loss.png")

# 7.2 Scatter Real × Predito
fig, ax = plt.subplots(figsize=(8, 8))
lims = [min(y_true.min(), y_pred.min()) - 10, max(y_true.max(), y_pred.max()) + 10]
ax.scatter(y_true, y_pred, alpha=0.45, s=18, color="#7C3AED", edgecolors="none")
ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x (perfeito)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"CNN Multi-View — Real × Predito  |  R² = {r2:.4f}", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Predito (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "cnn_multiview_dispersao_r2.png"), dpi=DPI)
plt.close(fig)
print("  [OK] cnn_multiview_dispersao_r2.png")

# 7.3 Histograma de Erros
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=50, color="#7C3AED", edgecolor="white", alpha=0.85)
ax.axvline(0, color="black", linewidth=1.8, linestyle="--", label="Erro = 0")
ax.axvline(errors.mean(), color="#DC2626", linewidth=1.8,
           label=f"Média: {errors.mean():.1f} kg")
ax.set_title("CNN Multi-View — Distribuição dos Erros", fontsize=14, fontweight="bold")
ax.set_xlabel("Erro (Predito − Real) [kg]"); ax.set_ylabel("Frequência")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "cnn_multiview_histograma_erros.png"), dpi=DPI)
plt.close(fig)
print("  [OK] cnn_multiview_histograma_erros.png")

# 7.4 Correlação Linear
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_true, y_pred, alpha=0.4, s=18, color="#7C3AED", edgecolors="none", label="Amostras")
coef  = np.polyfit(y_true, y_pred, 1)
xline = np.linspace(y_true.min(), y_true.max(), 100)
ax.plot(xline, np.polyval(coef, xline), color="#DC2626", linewidth=2,
        label=f"Tendência (r = {pearson_r:.4f})")
ax.plot(xline, xline, "k--", linewidth=1.2, alpha=0.5, label="y = x (perfeito)")
ax.set_title(f"CNN Multi-View — Correlação Linear\nPearson r = {pearson_r:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Predito (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "cnn_multiview_correlacao_linear.png"), dpi=DPI)
plt.close(fig)
print("  [OK] cnn_multiview_correlacao_linear.png")

# 8. SALVAR RESULTADOS
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
np.save(safe_path(os.path.join(SAVE_DIR, "predictions_cnn_multiview.npy")), np.array([y_true, y_pred]))
salvar_json(RESULTS_PATH, "CNN Multi-View", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
    "train_history": history["train_loss"], "val_history": history["val_loss"],
})

print(f"\n  Modelo salvo : resnet18_multiview_gado.pth  (época {best_epoch})")
print(f"  Resultados   : resultados_modelos.json")
print(f"  Gráficos     : graficos/")
