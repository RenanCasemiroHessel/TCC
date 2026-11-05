"""
CNN Multi-View (ResNet18 Siamesa) com K-Fold Cross-Validation — TCC2

Extensão do melhor modelo do TCC1 (MAE 20.75kg, single split 80/20) para:
  1. Faixa de peso estendida (30-1000kg, incluindo os 46 pares >=250kg antes
     descartados pelo corte <250kg do TCC1).
  2. K-Fold (k=K_FOLDS) por animal, sem vazamento de dados entre folds.
  3. Sobre-amostragem da classe "pesado" (>=250kg, só 1.9% do dataset) dentro
     de cada fold de treino, para o modelo não ignorá-la estatisticamente.
  4. Auditoria de erro por faixa de peso (Bezerro/Adulto/Boi Gordo) usando
     predições out-of-fold — cada animal é avaliado exatamente uma vez, por
     um modelo que nunca o viu no treino.

ATENÇÃO — tempo de execução:
  NUM_EPOCHS e K_FOLDS controlam o custo total. Uma rodada única de 30 épocas
  do multi-view no TCC1 levou ~2-3h numa RTX 4050. K_FOLDS=5 nessa mesma
  escala é ~5x isso. Os valores abaixo começam reduzidos (execução rápida de
  validação) — ajuste antes de rodar a versão "final" para o relatório.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                      "xtick.labelsize": 13, "ytick.labelsize": 13,
                      "legend.fontsize": 12, "figure.titlesize": 17})
import time
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_tcc import safe_path, salvar_json

# ── Configuração ─────────────────────────────────────────────────────────
SEED            = 42
IMG_SIZE        = 224
BATCH_SIZE      = 16
NUM_EPOCHS      = 15      # ~640s/época medido no run K=3x8 -> K=3x15 ~8h
K_FOLDS         = 3
FINETUNE_EPOCH  = 3
AUG_MULTIPLIER  = 4        # multiplicador geral (igual TCC1)
PESADO_LIMIAR   = 250.0    # >= isso é "pesado" (raro, precisa reforço)
PESADO_OVERSAMPLE = 16     # multiplicador EXTRA só p/ os pares pesados no treino

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR    = os.path.join(SAVE_DIR, "..", "..", "dados")
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")   # pasta graficos/ já existente do TCC1
MODELOS_DIR  = SAVE_DIR   # .pth soltos na raiz, mesmo padrão do TCC1
os.makedirs(GRAFICOS_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

print(f"  CNN Multi-View K-Fold — TCC2 (faixa estendida)")
print(f"  Device      : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"  Épocas/fold : {NUM_EPOCHS}")
print(f"  K-Folds     : {K_FOLDS}")
print(f"  Batch size  : {BATCH_SIZE}")

csv_path = os.path.join(DADOS_DIR, "dataset_pares_estendido.csv")
if not os.path.exists(csv_path):
    print(f"\n  ERRO: {csv_path} não encontrado.")
    print("  Rode primeiro: python preparar_dataset_estendido.py")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"\n  Dataset: {len(df)} pares | faixa {df['weight'].min():.0f}-{df['weight'].max():.0f}kg")
pesados_mask = df["weight"] >= PESADO_LIMIAR
print(f"  Pesados (>={PESADO_LIMIAR:.0f}kg): {pesados_mask.sum()} pares ({pesados_mask.mean()*100:.1f}%)")


def cor_mascara(img_path):
    return np.array([0, 255, 0, 255]) if "B2" in img_path else np.array([255, 30, 249, 255])


def load_segmented(img_path, mask_path):
    try:
        img_np = np.array(Image.open(img_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path))
        if mask_np.shape[-1] == 3:
            alpha = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
            mask_np = np.concatenate([mask_np, alpha], axis=-1)
        cor = cor_mascara(img_path)
        mask = np.all(mask_np == cor, axis=-1)
        out = img_np.copy()
        out[~mask] = 0
        return Image.fromarray(out)
    except Exception:
        return Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))


class MultiViewDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        side_img = load_segmented(row["side_path"], row["side_mask"])
        rear_img = load_segmented(row["rear_path"], row["rear_mask"])
        weight = torch.tensor(row["weight"], dtype=torch.float32)
        if self.transform:
            side_img = self.transform(side_img)
            rear_img = self.transform(rear_img)
        return side_img, rear_img, weight


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


class SiameseResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x_side, x_rear):
        feat_side = self.backbone(x_side).flatten(1)
        feat_rear = self.backbone(x_rear).flatten(1)
        feat = torch.cat([feat_side, feat_rear], dim=1)
        return self.head(feat)


def train_one_fold(fold_idx, train_df, val_df):
    train_dataset_base = MultiViewDataset(train_df, train_transform)
    val_dataset = MultiViewDataset(val_df, val_transform)

    # oversampling: duplica os pares pesados extra vezes dentro do treino,
    # além do AUG_MULTIPLIER geral que todo mundo já recebe.
    pesados_train = train_df[train_df["weight"] >= PESADO_LIMIAR]
    pesados_dataset = MultiViewDataset(pesados_train, train_transform) if len(pesados_train) else None

    partes = [train_dataset_base] * AUG_MULTIPLIER
    if pesados_dataset is not None and len(pesados_dataset) > 0:
        partes += [pesados_dataset] * PESADO_OVERSAMPLE
    train_dataset = ConcatDataset(partes)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=torch.cuda.is_available())

    Y_MIN = float(train_df["weight"].min())
    Y_MAX = float(train_df["weight"].max())

    model = SiameseResNet().to(DEVICE)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    huber_delta = 50.0 / (Y_MAX - Y_MIN)
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    model_path = safe_path(os.path.join(MODELOS_DIR, f"resnet18_multiview_estendido_fold{fold_idx}.pth"))

    print(f"\n  ── Fold {fold_idx}/{K_FOLDS} ── treino={len(train_df)} pares "
          f"({len(train_dataset)} efetivos/época) | val={len(val_df)} pares")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        if epoch == FINETUNE_EPOCH:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

        model.train()
        running = 0.0
        for side, rear, weights in train_loader:
            side, rear = side.to(DEVICE), rear.to(DEVICE)
            w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(side, rear)
            loss = criterion(preds, w_norm)
            loss.backward()
            optimizer.step()
            running += loss.item() * side.size(0)
        train_loss = running / len(train_dataset)

        model.eval()
        val_run = 0.0
        with torch.no_grad():
            for side, rear, weights in val_loader:
                side, rear = side.to(DEVICE), rear.to(DEVICE)
                w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
                preds = model(side, rear)
                val_run += criterion(preds, w_norm).item() * side.size(0)
        val_loss = val_run / len(val_dataset)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "y_min": Y_MIN, "y_max": Y_MAX, "img_size": IMG_SIZE}, model_path)
            marker = "  * melhor"

        print(f"    Época {epoch:02d}/{NUM_EPOCHS} | Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | {time.time()-t0:.1f}s{marker}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds_kg, true_kg = [], []
    with torch.no_grad():
        for side, rear, weights in val_loader:
            side, rear = side.to(DEVICE), rear.to(DEVICE)
            preds = model(side, rear).squeeze(1).cpu().numpy()
            preds_kg.extend((preds * (Y_MAX - Y_MIN) + Y_MIN).tolist())
            true_kg.extend(weights.numpy().tolist())

    return {
        "history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "y_true": true_kg, "y_pred": preds_kg, "model_path": model_path,
    }


# ── K-Fold por animal (sem vazamento) ───────────────────────────────────
keys = df["key"].unique()
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
oof_true, oof_pred = [], []   # out-of-fold: cada animal avaliado 1x, por modelo que nunca o viu

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(keys), start=1):
    train_keys = keys[train_idx]
    val_keys = keys[val_idx]
    train_df = df[df["key"].isin(train_keys)].reset_index(drop=True)
    val_df = df[df["key"].isin(val_keys)].reset_index(drop=True)

    result = train_one_fold(fold_idx, train_df, val_df)
    fold_results.append(result)
    oof_true.extend(result["y_true"])
    oof_pred.extend(result["y_pred"])

# ── Métricas agregadas ───────────────────────────────────────────────────
per_fold_mae, per_fold_rmse, per_fold_r2 = [], [], []
for r in fold_results:
    yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
    per_fold_mae.append(mean_absolute_error(yt, yp))
    per_fold_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
    per_fold_r2.append(r2_score(yt, yp))

oof_true = np.array(oof_true)
oof_pred = np.array(oof_pred)
oof_mae = mean_absolute_error(oof_true, oof_pred)
oof_rmse = np.sqrt(mean_squared_error(oof_true, oof_pred))
oof_r2 = r2_score(oof_true, oof_pred)
oof_errors = oof_pred - oof_true

print(f"""
{'='*60}
  RESULTADO K-FOLD ({K_FOLDS} folds) — CNN Multi-View, faixa estendida
{'='*60}

  Por fold:
""")
for i, (mae, rmse, r2) in enumerate(zip(per_fold_mae, per_fold_rmse, per_fold_r2), start=1):
    print(f"    Fold {i}: MAE={mae:6.2f}kg  RMSE={rmse:6.2f}kg  R²={r2:.4f}")

print(f"""
  Média ± desvio padrão entre folds:
    MAE  : {np.mean(per_fold_mae):.2f} ± {np.std(per_fold_mae):.2f} kg
    RMSE : {np.mean(per_fold_rmse):.2f} ± {np.std(per_fold_rmse):.2f} kg
    R²   : {np.mean(per_fold_r2):.4f} ± {np.std(per_fold_r2):.4f}

  Out-of-fold (todo o dataset, cada animal avaliado 1x por modelo que nunca o viu):
    MAE  : {oof_mae:.2f} kg
    RMSE : {oof_rmse:.2f} kg
    R²   : {oof_r2:.4f}
""")

# ── Gráficos ──────────────────────────────────────────────────────────────
DPI = 150

# Curvas de loss (uma linha por fold)
fig, ax = plt.subplots(figsize=(9, 5))
cores = plt.cm.viridis(np.linspace(0, 0.85, K_FOLDS))
for i, r in enumerate(fold_results):
    ep = range(1, NUM_EPOCHS + 1)
    ax.plot(ep, r["history"]["val_loss"], "-", color=cores[i], linewidth=2, label=f"Fold {i+1} (val)")
ax.set_title(f"CNN Multi-View — Val Loss por Fold (K={K_FOLDS})", fontsize=14, fontweight="bold")
ax.set_xlabel("Época"); ax.set_ylabel("Huber Loss (normalizado)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_curvas_loss.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_curvas_loss.png")

# Violin plot de erro por faixa de peso (auditoria — Tabela 8 do plano)
faixas = [(30, 200, "Bezerro\n(<200kg)"), (200, 450, "Adulto\n(200-450kg)"), (450, 1000, "Boi Gordo\n(>450kg)")]
labels_faixa = []
for v, e in zip(oof_true, oof_errors):
    nome = next((nome for lo, hi, nome in faixas if lo <= v < hi), "?")
    labels_faixa.append(nome)

audit_df = pd.DataFrame({"faixa": labels_faixa, "erro": oof_errors})
fig, ax = plt.subplots(figsize=(9, 6))
ordem = [nome for _, _, nome in faixas if nome in audit_df["faixa"].values]
sns.violinplot(data=audit_df, x="faixa", y="erro", order=ordem, ax=ax, palette="Set2", cut=0)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
ax.set_title("Auditoria de Erro por Faixa de Peso (out-of-fold)", fontsize=14, fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("Erro (Predito − Real) [kg]")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_violin_erro_por_faixa.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_violin_erro_por_faixa.png")

# Dispersão real x predito (out-of-fold)
fig, ax = plt.subplots(figsize=(8, 8))
lims = [min(oof_true.min(), oof_pred.min()) - 10, max(oof_true.max(), oof_pred.max()) + 10]
ax.scatter(oof_true, oof_pred, alpha=0.4, s=18, color="#7C3AED", edgecolors="none")
ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x (perfeito)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"CNN Multi-View K-Fold — Real × Predito (out-of-fold)\nR² = {oof_r2:.4f}", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Predito (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_dispersao_oof.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_dispersao_oof.png")

# ── Persistência ──────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(DADOS_DIR, "resultados_kfold.json")
np.save(safe_path(os.path.join(DADOS_DIR, "predictions_kfold_oof.npy")), np.array([oof_true, oof_pred]))
salvar_json(RESULTS_PATH, "CNN Multi-View K-Fold (estendido)", {
    "k_folds": K_FOLDS,
    "num_epochs": NUM_EPOCHS,
    "per_fold_mae": per_fold_mae, "per_fold_rmse": per_fold_rmse, "per_fold_r2": per_fold_r2,
    "mae_mean": float(np.mean(per_fold_mae)), "mae_std": float(np.std(per_fold_mae)),
    "rmse_mean": float(np.mean(per_fold_rmse)), "rmse_std": float(np.std(per_fold_rmse)),
    "r2_mean": float(np.mean(per_fold_r2)), "r2_std": float(np.std(per_fold_r2)),
    "oof_mae": float(oof_mae), "oof_rmse": float(oof_rmse), "oof_r2": float(oof_r2),
})

print(f"\n  Resultados salvos em: {RESULTS_PATH}")
print(f"  Modelos por fold em : {MODELOS_DIR}/")
print(f"  Gráficos em          : {GRAFICOS_DIR}/")
