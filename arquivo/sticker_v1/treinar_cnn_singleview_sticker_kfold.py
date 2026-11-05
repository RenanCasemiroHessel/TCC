"""
CNN Single-View calibrada pelo sticker — com K-Fold e retomada automática.

Mesma ideia do treinar_cnn_singleview_sticker.py (zoom proporcional ao
tamanho do sticker + sticker_ratio injetado como escalar na rede), agora em
K-Fold, seguindo o mesmo padrão do treinar_cnn_singleview_kfold.py original.

Checkpoint e retomada:
  Cada fold salva DOIS arquivos:
    - resnet18_singleview_sticker_fold{N}.pth         -> melhor época (val loss mínimo)
    - resnet18_singleview_sticker_fold{N}_latest.pth  -> última época treinada,
      com otimizador, histórico e contador de época — usado só pra retomar.

  Ao rodar o script de novo:
    - Se o fold já tem {N} épocas treinadas (latest.epoch >= NUM_EPOCHS):
      não retreina, só recarrega o melhor checkpoint e reavalia no val set.
    - Se tem menos épocas que NUM_EPOCHS: retoma exatamente da próxima época,
      com otimizador/scheduler/congelamento no estado certo pra aquele ponto.
    - Se não existe checkpoint nenhum: começa do zero.

  "Polir mais": se um fold já terminou e você aumenta NUM_EPOCHS aqui em
  cima e roda de novo, ele detecta que start_epoch <= NUM_EPOCHS e continua
  treinando esse fold a partir de onde parou, com o cosine schedule
  recalculado pro novo total de épocas.

  Se o treino travar no meio (CUDA error, queda de energia etc.), é só rodar
  o script de novo — ele resume sozinho a partir do '_latest' de cada fold.
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
import seaborn as sns
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                      "xtick.labelsize": 13, "ytick.labelsize": 13,
                      "legend.fontsize": 12, "figure.titlesize": 17})
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_tcc import salvar_json

# ── Configuração ──────────────────────────────────────────────────────────
SEED            = 42
IMG_SIZE        = 224
CANVAS_PX       = 2200     # medido nos dados: largura zoom p99=1945px, max=2197px
BATCH_SIZE      = 32
NUM_EPOCHS      = 10       # aumente aqui + rode de novo pra "polir" um fold já terminado
K_FOLDS         = 5        # mesmo K do treinar_cnn_singleview_kfold.py original
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

CORES = {
    "B2": {"cattle": (0, 255, 0), "sticker": (255, 240, 0)},
    "B4": {"cattle": (255, 30, 249), "sticker": (0, 117, 255)},
}

print(f"  CNN Single-View calibrada pelo sticker — K-Fold")
print(f"  Device      : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"  Épocas/fold : {NUM_EPOCHS}")
print(f"  K-Folds     : {K_FOLDS}")

CSV_PATH = os.path.join(SAVE_DIR, "dataset_features_sticker.csv")
if not os.path.exists(CSV_PATH):
    print(f"\n  ERRO: {CSV_PATH} não encontrado.")
    print("  Rode primeiro: python extrair_features_sticker.py")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df = df[df["vista"] == "side"].reset_index(drop=True)  # sticker só existe em lateral

REF_STICKER_DIAM = float(np.sqrt(df["sticker_area"]).median())
print(f"  Amostras (lateral, com sticker) : {len(df)}")
print(f"  Diâmetro de referência do sticker: {REF_STICKER_DIAM:.1f}px")


def carregar_escalado(img_path, mask_path, lote, sticker_area):
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
        return img_pil.crop((left, top, left + CANVAS_PX, top + CANVAS_PX))
    except Exception:
        return Image.fromarray(np.zeros((CANVAS_PX, CANVAS_PX, 3), dtype=np.uint8))


class StickerScaledDataset(Dataset):
    def __init__(self, dataframe, ratio_mean, ratio_std, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.ratio_mean = ratio_mean
        self.ratio_std = ratio_std
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = carregar_escalado(row["path"], row["mask"], row["lote"], row["sticker_area"])
        if self.transform:
            img = self.transform(img)
        ratio_norm = (np.log(row["sticker_ratio"]) - self.ratio_mean) / (self.ratio_std + 1e-8)
        ratio = torch.tensor([ratio_norm], dtype=torch.float32)
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


def acc10(yt, yp):
    return float((np.abs(yp - yt) / yt <= 0.10).mean() * 100)


@torch.no_grad()
def avaliar(model, val_loader, Y_MIN, Y_MAX):
    model.eval()
    preds_kg, true_kg = [], []
    for imgs, ratios, weights in val_loader:
        imgs, ratios = imgs.to(DEVICE), ratios.to(DEVICE)
        preds = model(imgs, ratios).squeeze(1).cpu().numpy()
        preds_kg.extend((preds * (Y_MAX - Y_MIN) + Y_MIN).tolist())
        true_kg.extend(weights.numpy().tolist())
    return preds_kg, true_kg


def treinar_ou_retomar_fold(fold_idx, train_df, val_df):
    best_path = os.path.join(MODELOS_DIR, f"resnet18_singleview_sticker_fold{fold_idx}.pth")
    latest_path = os.path.join(MODELOS_DIR, f"resnet18_singleview_sticker_fold{fold_idx}_latest.pth")

    Y_MIN = float(train_df["peso_kg"].min())
    Y_MAX = float(train_df["peso_kg"].max())
    log_ratio_train = np.log(train_df["sticker_ratio"].values)
    ratio_mean = float(log_ratio_train.mean())
    ratio_std = float(log_ratio_train.std())

    val_dataset = StickerScaledDataset(val_df, ratio_mean, ratio_std, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=torch.cuda.is_available())

    model = StickerScaledNet().to(DEVICE)
    huber_delta = 50.0 / (Y_MAX - Y_MIN)
    criterion = nn.HuberLoss(delta=huber_delta)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    ck_latest = None

    if os.path.exists(latest_path):
        ck_latest = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(ck_latest["model_state_dict"])
        history = ck_latest["history"]
        best_val_loss = ck_latest["best_val_loss"]
        best_epoch = ck_latest["best_epoch"]
        start_epoch = ck_latest["epoch"] + 1
        print(f"\n  ── Fold {fold_idx}/{K_FOLDS} ── RETOMANDO da época {start_epoch} "
              f"(checkpoint tinha {ck_latest['epoch']} época(s))")
    else:
        print(f"\n  ── Fold {fold_idx}/{K_FOLDS} ── começando do zero | "
              f"treino={len(train_df)} imgs | val={len(val_df)} imgs")

    if start_epoch > NUM_EPOCHS:
        print(f"    já tem {start_epoch - 1} épocas (>= NUM_EPOCHS={NUM_EPOCHS}) — só reavaliando")
        checkpoint = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        preds_kg, true_kg = avaliar(model, val_loader, Y_MIN, Y_MAX)
        return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                "y_true": true_kg, "y_pred": preds_kg}

    ja_descongelado = start_epoch >= FINETUNE_EPOCH
    for name, param in model.named_parameters():
        param.requires_grad = True if ja_descongelado else ("backbone" not in name)
    lr_inicial = 1e-5 if ja_descongelado else 1e-4
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr_inicial, weight_decay=1e-4)
    if ck_latest is not None and "optimizer_state_dict" in ck_latest:
        try:
            optimizer.load_state_dict(ck_latest["optimizer_state_dict"])
        except Exception:
            print("    [!] estado do otimizador incompatível (provável troca de congelamento) — "
                  "retomando com otimizador novo, pesos do modelo preservados")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    for _ in range(start_epoch - 1):
        scheduler.step()

    train_dataset_base = StickerScaledDataset(train_df, ratio_mean, ratio_std, train_transform)
    train_dataset = ConcatDataset([train_dataset_base] * AUG_MULTIPLIER)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=torch.cuda.is_available())

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()
        if epoch == FINETUNE_EPOCH and not ja_descongelado:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
            ja_descongelado = True

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
                        "ratio_mean": ratio_mean, "ratio_std": ratio_std}, best_path)
            marker = "  * melhor"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }, latest_path)

        print(f"    Época {epoch:02d}/{NUM_EPOCHS} | Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | {time.time()-t0:.1f}s{marker}")

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    preds_kg, true_kg = avaliar(model, val_loader, Y_MIN, Y_MAX)

    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "y_true": true_kg, "y_pred": preds_kg}


# ── K-Fold por animal ──────────────────────────────────────────────────────
keys = df["id_animal"].unique()
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
oof_true, oof_pred = [], []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(keys), start=1):
    train_keys = keys[train_idx]
    val_keys = keys[val_idx]
    train_df = df[df["id_animal"].isin(train_keys)].reset_index(drop=True)
    val_df = df[df["id_animal"].isin(val_keys)].reset_index(drop=True)

    result = treinar_ou_retomar_fold(fold_idx, train_df, val_df)
    fold_results.append(result)
    oof_true.extend(result["y_true"])
    oof_pred.extend(result["y_pred"])

# ── Métricas agregadas ───────────────────────────────────────────────────
per_fold_mae, per_fold_rmse, per_fold_r2, per_fold_acc10 = [], [], [], []
for r in fold_results:
    yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
    per_fold_mae.append(mean_absolute_error(yt, yp))
    per_fold_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
    per_fold_r2.append(r2_score(yt, yp))
    per_fold_acc10.append(acc10(yt, yp))

oof_true = np.array(oof_true)
oof_pred = np.array(oof_pred)
oof_mae = mean_absolute_error(oof_true, oof_pred)
oof_rmse = np.sqrt(mean_squared_error(oof_true, oof_pred))
oof_r2 = r2_score(oof_true, oof_pred)
oof_pearson_r = float(np.corrcoef(oof_true, oof_pred)[0, 1])
oof_acc10 = acc10(oof_true, oof_pred)

print(f"""
{'='*60}
  RESULTADO K-FOLD ({K_FOLDS} folds) — CNN Single-View + sticker
{'='*60}

  Por fold:
""")
for i, (mae, rmse, r2, a10) in enumerate(zip(per_fold_mae, per_fold_rmse, per_fold_r2, per_fold_acc10), start=1):
    print(f"    Fold {i}: MAE={mae:6.2f}kg  RMSE={rmse:6.2f}kg  R²={r2:.4f}  Acc±10%={a10:.1f}%")

print(f"""
  Média ± desvio padrão entre folds:
    MAE      : {np.mean(per_fold_mae):.2f} ± {np.std(per_fold_mae):.2f} kg
    RMSE     : {np.mean(per_fold_rmse):.2f} ± {np.std(per_fold_rmse):.2f} kg
    R²       : {np.mean(per_fold_r2):.4f} ± {np.std(per_fold_r2):.4f}

  Out-of-fold (geral):
    MAE      : {oof_mae:.2f} kg
    RMSE     : {oof_rmse:.2f} kg
    R²       : {oof_r2:.4f}
    Pearson r: {oof_pearson_r:.4f}
    Acc ±10% : {oof_acc10:.1f} %

  Comparar com:
    CNN Single-View K-Fold original (sem sticker): MAE 22,95kg | R² 0,3653 (3 folds, dataset estendido)
    LR tabular + sticker_ratio (split único)      : MAE 21,03kg | R² 0,4058
""")

# ── Gráficos ──────────────────────────────────────────────────────────────
DPI = 150
fig, ax = plt.subplots(figsize=(9, 5))
cores_plot = plt.cm.viridis(np.linspace(0, 0.85, K_FOLDS))
for i, r in enumerate(fold_results):
    ep = range(1, len(r["history"]["val_loss"]) + 1)
    ax.plot(ep, r["history"]["val_loss"], "-", color=cores_plot[i], linewidth=2, label=f"Fold {i+1} (val)")
ax.set_title(f"CNN Single-View + sticker — Val Loss por Fold (K={K_FOLDS})", fontsize=14, fontweight="bold")
ax.set_xlabel("Época"); ax.set_ylabel("Huber Loss (normalizado)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_singleview_sticker_curvas_loss.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_singleview_sticker_curvas_loss.png")

fig, ax = plt.subplots(figsize=(8, 8))
lims = [min(oof_true.min(), oof_pred.min()) - 10, max(oof_true.max(), oof_pred.max()) + 10]
ax.scatter(oof_true, oof_pred, alpha=0.4, s=18, color="#2563EB", edgecolors="none")
ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x (perfeito)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"CNN Single-View + sticker — Real × Predito (OOF)\nR² = {oof_r2:.4f}", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Predito (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_singleview_sticker_dispersao_oof.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_singleview_sticker_dispersao_oof.png")

# ── Persistência ────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_kfold_singleview_sticker.json")
np.save(os.path.join(SAVE_DIR, "predictions_kfold_singleview_sticker_oof.npy"), np.array([oof_true, oof_pred]))
salvar_json(RESULTS_PATH, "CNN Single-View K-Fold + sticker (zoom + escalar)", {
    "k_folds": K_FOLDS, "num_epochs": NUM_EPOCHS,
    "per_fold_mae": per_fold_mae, "per_fold_rmse": per_fold_rmse,
    "per_fold_r2": per_fold_r2, "per_fold_acc10": per_fold_acc10,
    "mae_mean": float(np.mean(per_fold_mae)), "mae_std": float(np.std(per_fold_mae)),
    "rmse_mean": float(np.mean(per_fold_rmse)), "rmse_std": float(np.std(per_fold_rmse)),
    "r2_mean": float(np.mean(per_fold_r2)), "r2_std": float(np.std(per_fold_r2)),
    "oof_mae": float(oof_mae), "oof_rmse": float(oof_rmse), "oof_r2": float(oof_r2),
    "oof_pearson_r": oof_pearson_r, "oof_acc10": oof_acc10,
    "ref_sticker_diam_px": REF_STICKER_DIAM, "canvas_px": CANVAS_PX,
})

print(f"\n  Resultados salvos em: {RESULTS_PATH}")
print(f"  Modelos por fold em : {MODELOS_DIR}/")
print(f"  Gráficos em          : {GRAFICOS_DIR}/")
