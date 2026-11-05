"""
CNN Single-View (ResNet18) com K-Fold Cross-Validation — TCC2

Combina duas fontes:
  1. dataset_pares_estendido.csv  — os pares side+rear (B2+B4), desmembrados
     em duas amostras independentes cada (lateral e traseira).
  2. dataset_singleview_b3.csv    — extra do lote B3 (só lateral, 2.585
     imagens, 99 delas >=250kg, até 621kg). Ver preparar_dataset_estendido.py
     para o porquê do B3 só entrar aqui e não no multi-view.

Por causa do B3, o K-Fold aqui NÃO usa exatamente o mesmo split de animais
do script multi-view (que só conhece os pares) — os animais compartilhados
(B2+B4) ainda caem nos mesmos folds pela mesma seed, mas o B3 entra com
chaves extras só neste script. Isso é intencional: o objetivo agora é
aproveitar o máximo de dado pesado disponível, não mais uma comparação
1:1 perfeita com o multi-view.

ATENÇÃO — tempo de execução: mesma ordem de grandeza do multi-view (ver
comentário no topo de treinar_cnn_multiview_kfold.py). Com o B3 incluído, o
dataset é maior — espere um pouco mais por época.
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
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_tcc import safe_path, salvar_json

# ── Configuração (mesmos valores do script multi-view, para comparação justa) ──
SEED               = 42
IMG_SIZE           = 224
BATCH_SIZE         = 32     # single-view: 1 imagem/amostra -> pode dobrar o batch do multi-view
NUM_EPOCHS         = 30
K_FOLDS            = 5
FINETUNE_EPOCH     = 3
AUG_MULTIPLIER     = 4
PESADO_LIMIAR      = 250.0
PESADO_OVERSAMPLE  = 16

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

print(f"  CNN Single-View K-Fold — TCC2 (faixa estendida)")
print(f"  Device      : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"  Épocas/fold : {NUM_EPOCHS}")
print(f"  K-Folds     : {K_FOLDS}")
print(f"  Batch size  : {BATCH_SIZE}")

csv_path = os.path.join(DADOS_DIR, "dataset_pares_estendido.csv")
csv_b3_path = os.path.join(DADOS_DIR, "dataset_singleview_b3.csv")
if not os.path.exists(csv_path):
    print(f"\n  ERRO: {csv_path} não encontrado.")
    print("  Rode primeiro: python preparar_dataset_estendido.py")
    sys.exit(1)

pares = pd.read_csv(csv_path)

# desmembra cada par (side+rear) em duas amostras single-view independentes,
# preservando "key" (animal) para o K-Fold não vazar dados entre treino/val
linhas = []
for _, row in pares.iterrows():
    linhas.append({"key": row["key"], "path": row["side_path"], "mask": row["side_mask"],
                    "weight": row["weight"], "lote": row["lote"], "vista": "side"})
    linhas.append({"key": row["key"], "path": row["rear_path"], "mask": row["rear_mask"],
                    "weight": row["weight"], "lote": row["lote"], "vista": "rear"})
df = pd.DataFrame(linhas)
n_pares_b2b4 = len(pares)

# extra do B3 (só lateral, sem par) — ver preparar_dataset_estendido.py
if os.path.exists(csv_b3_path):
    df_b3 = pd.read_csv(csv_b3_path)
    df = pd.concat([df, df_b3[["key", "path", "mask", "weight", "lote", "vista"]]], ignore_index=True)
    print(f"\n  B3 incorporado: +{len(df_b3)} imagens (só lateral)")
else:
    print(f"\n  [!] {csv_b3_path} não encontrado — rode preparar_dataset_estendido.py p/ incluir o B3")

print(f"  Dataset total: {len(df)} imagens ({n_pares_b2b4} pares B2/B4 + B3 avulso) | faixa {df['weight'].min():.0f}-{df['weight'].max():.0f}kg")
pesados_mask = df["weight"] >= PESADO_LIMIAR
print(f"  Pesados (>={PESADO_LIMIAR:.0f}kg): {pesados_mask.sum()} imagens ({pesados_mask.mean()*100:.1f}%)")


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


class SingleViewDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_segmented(row["path"], row["mask"])
        weight = torch.tensor(row["weight"], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, weight


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


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 1), nn.Sigmoid(),
    )
    return model


def train_one_fold(fold_idx, train_df, val_df):
    train_dataset_base = SingleViewDataset(train_df, train_transform)
    val_dataset = SingleViewDataset(val_df, val_transform)

    pesados_train = train_df[train_df["weight"] >= PESADO_LIMIAR]
    pesados_dataset = SingleViewDataset(pesados_train, train_transform) if len(pesados_train) else None

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

    model = build_model().to(DEVICE)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    huber_delta = 50.0 / (Y_MAX - Y_MIN)
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    model_path = safe_path(os.path.join(MODELOS_DIR, f"resnet18_singleview_estendido_fold{fold_idx}.pth"))

    print(f"\n  ── Fold {fold_idx}/{K_FOLDS} ── treino={len(train_df)} imgs "
          f"({len(train_dataset)} efetivas/época) | val={len(val_df)} imgs")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        if epoch == FINETUNE_EPOCH:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

        model.train()
        running = 0.0
        for imgs, weights in train_loader:
            imgs = imgs.to(DEVICE)
            w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, w_norm)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_dataset)

        model.eval()
        val_run = 0.0
        with torch.no_grad():
            for imgs, weights in val_loader:
                imgs = imgs.to(DEVICE)
                w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
                preds = model(imgs)
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
                        "y_min": Y_MIN, "y_max": Y_MAX, "img_size": IMG_SIZE}, model_path)
            marker = "  * melhor"

        print(f"    Época {epoch:02d}/{NUM_EPOCHS} | Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | {time.time()-t0:.1f}s{marker}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds_kg, true_kg = [], []
    with torch.no_grad():
        for imgs, weights in val_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).squeeze(1).cpu().numpy()
            preds_kg.extend((preds * (Y_MAX - Y_MIN) + Y_MIN).tolist())
            true_kg.extend(weights.numpy().tolist())

    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "y_true": true_kg, "y_pred": preds_kg, "model_path": model_path}


# ── K-Fold por animal ─────────────────────────────────────────────────
# keys vêm do df completo (pares B2/B4 + avulso B3), não só dos pares —
# os animais compartilhados com o multi-view (B2/B4) ainda caem no mesmo
# fold pela mesma seed, mas o B3 (só aqui) entra com chaves próprias.
keys = df["key"].unique()
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
oof_true, oof_pred = [], []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(keys), start=1):
    train_keys = keys[train_idx]
    val_keys = keys[val_idx]
    train_df = df[df["key"].isin(train_keys)].reset_index(drop=True)
    val_df = df[df["key"].isin(val_keys)].reset_index(drop=True)

    result = train_one_fold(fold_idx, train_df, val_df)
    fold_results.append(result)
    oof_true.extend(result["y_true"])
    oof_pred.extend(result["y_pred"])

def acc10(yt, yp):
    """% de predições com erro absoluto <= 10% do peso real (mesma métrica do TCC1)."""
    return float((np.abs(yp - yt) / yt <= 0.10).mean() * 100)


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
oof_acc10 = acc10(oof_true, oof_pred)
oof_pearson_r = float(np.corrcoef(oof_true, oof_pred)[0, 1])
oof_errors = oof_pred - oof_true

# mesma quebra por faixa que usamos manualmente antes — agora automática
mask_leve_oof = oof_true < PESADO_LIMIAR
mask_pesado_oof = oof_true >= PESADO_LIMIAR
mae_leve = mean_absolute_error(oof_true[mask_leve_oof], oof_pred[mask_leve_oof]) if mask_leve_oof.any() else float("nan")
mae_pesado = mean_absolute_error(oof_true[mask_pesado_oof], oof_pred[mask_pesado_oof]) if mask_pesado_oof.any() else float("nan")
acc10_leve = acc10(oof_true[mask_leve_oof], oof_pred[mask_leve_oof]) if mask_leve_oof.any() else float("nan")
acc10_pesado = acc10(oof_true[mask_pesado_oof], oof_pred[mask_pesado_oof]) if mask_pesado_oof.any() else float("nan")

print(f"""
{'='*60}
  RESULTADO K-FOLD ({K_FOLDS} folds) — CNN Single-View, faixa estendida
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
    Acc ±10% : {np.mean(per_fold_acc10):.1f} ± {np.std(per_fold_acc10):.1f} %

  Out-of-fold (geral):
    MAE      : {oof_mae:.2f} kg
    RMSE     : {oof_rmse:.2f} kg
    R²       : {oof_r2:.4f}
    Pearson r: {oof_pearson_r:.4f}
    Acc ±10% : {oof_acc10:.1f} %

  Out-of-fold, por faixa (corte em {PESADO_LIMIAR:.0f}kg):
    <{PESADO_LIMIAR:.0f}kg  ({mask_leve_oof.sum():>4} amostras): MAE {mae_leve:6.2f}kg | Acc±10% {acc10_leve:5.1f}%
    >={PESADO_LIMIAR:.0f}kg ({mask_pesado_oof.sum():>4} amostras): MAE {mae_pesado:6.2f}kg | Acc±10% {acc10_pesado:5.1f}%
""")

# ── Gráficos ──────────────────────────────────────────────────────────────
DPI = 150

fig, ax = plt.subplots(figsize=(9, 5))
cores = plt.cm.viridis(np.linspace(0, 0.85, K_FOLDS))
for i, r in enumerate(fold_results):
    ep = range(1, NUM_EPOCHS + 1)
    ax.plot(ep, r["history"]["val_loss"], "-", color=cores[i], linewidth=2, label=f"Fold {i+1} (val)")
ax.set_title(f"CNN Single-View — Val Loss por Fold (K={K_FOLDS})", fontsize=14, fontweight="bold")
ax.set_xlabel("Época"); ax.set_ylabel("Huber Loss (normalizado)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_singleview_curvas_loss.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_singleview_curvas_loss.png")

faixas = [(30, 200, "Bezerro\n(<200kg)"), (200, 450, "Adulto\n(200-450kg)"), (450, 1000, "Boi Gordo\n(>450kg)")]
labels_faixa = []
for v in oof_true:
    nome = next((nome for lo, hi, nome in faixas if lo <= v < hi), "?")
    labels_faixa.append(nome)

audit_df = pd.DataFrame({"faixa": labels_faixa, "erro": oof_errors})
fig, ax = plt.subplots(figsize=(9, 6))
ordem = [nome for _, _, nome in faixas if nome in audit_df["faixa"].values]
sns.violinplot(data=audit_df, x="faixa", y="erro", order=ordem, ax=ax, palette="Set2", cut=0)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
ax.set_title("Single-View — Auditoria de Erro por Faixa de Peso (out-of-fold)", fontsize=14, fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("Erro (Predito − Real) [kg]")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_singleview_violin_erro_por_faixa.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_singleview_violin_erro_por_faixa.png")

fig, ax = plt.subplots(figsize=(8, 8))
lims = [min(oof_true.min(), oof_pred.min()) - 10, max(oof_true.max(), oof_pred.max()) + 10]
ax.scatter(oof_true, oof_pred, alpha=0.4, s=18, color="#2563EB", edgecolors="none")
ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x (perfeito)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"CNN Single-View K-Fold — Real × Predito (out-of-fold)\nR² = {oof_r2:.4f}", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Predito (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_singleview_dispersao_oof.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_singleview_dispersao_oof.png")

# histograma de erros (mesmo padrão dos gráficos do TCC1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(oof_errors, bins=50, color="#2563EB", edgecolor="white", alpha=0.85)
ax.axvline(0, color="black", linewidth=1.8, linestyle="--", label="Erro = 0")
ax.axvline(oof_errors.mean(), color="#DC2626", linewidth=1.8, label=f"Média: {oof_errors.mean():.1f} kg")
ax.set_title("CNN Single-View K-Fold — Distribuição dos Erros (out-of-fold)", fontsize=14, fontweight="bold")
ax.set_xlabel("Erro (Predito − Real) [kg]"); ax.set_ylabel("Frequência")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_singleview_histograma_erros.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_singleview_histograma_erros.png")

# correlação linear com Pearson r (mesmo padrão dos gráficos do TCC1)
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(oof_true, oof_pred, alpha=0.4, s=18, color="#2563EB", edgecolors="none", label="Amostras")
coef = np.polyfit(oof_true, oof_pred, 1)
xline = np.linspace(oof_true.min(), oof_true.max(), 100)
ax.plot(xline, np.polyval(coef, xline), color="#DC2626", linewidth=2, label=f"Tendência (r = {oof_pearson_r:.4f})")
ax.plot(xline, xline, "k--", linewidth=1.2, alpha=0.5, label="y = x (perfeito)")
ax.set_title(f"CNN Single-View K-Fold — Correlação Linear\nPearson r = {oof_pearson_r:.4f}", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Predito (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(GRAFICOS_DIR, "kfold_singleview_correlacao_linear.png"), dpi=DPI)
plt.close(fig)
print("  [OK] kfold_singleview_correlacao_linear.png")

# ── Persistência ──────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(DADOS_DIR, "resultados_kfold_singleview.json")
np.save(safe_path(os.path.join(DADOS_DIR, "predictions_kfold_singleview_oof.npy")), np.array([oof_true, oof_pred]))
salvar_json(RESULTS_PATH, "CNN Single-View K-Fold (estendido)", {
    "k_folds": K_FOLDS, "num_epochs": NUM_EPOCHS,
    "per_fold_mae": per_fold_mae, "per_fold_rmse": per_fold_rmse,
    "per_fold_r2": per_fold_r2, "per_fold_acc10": per_fold_acc10,
    "mae_mean": float(np.mean(per_fold_mae)), "mae_std": float(np.std(per_fold_mae)),
    "rmse_mean": float(np.mean(per_fold_rmse)), "rmse_std": float(np.std(per_fold_rmse)),
    "r2_mean": float(np.mean(per_fold_r2)), "r2_std": float(np.std(per_fold_r2)),
    "acc10_mean": float(np.mean(per_fold_acc10)), "acc10_std": float(np.std(per_fold_acc10)),
    "oof_mae": float(oof_mae), "oof_rmse": float(oof_rmse), "oof_r2": float(oof_r2),
    "oof_pearson_r": oof_pearson_r, "oof_acc10": oof_acc10,
    "oof_mae_leve": float(mae_leve), "oof_mae_pesado": float(mae_pesado),
    "oof_acc10_leve": float(acc10_leve), "oof_acc10_pesado": float(acc10_pesado),
    "pesado_limiar_kg": PESADO_LIMIAR,
})

print(f"\n  Resultados salvos em: {RESULTS_PATH}")
print(f"  Modelos por fold em : {MODELOS_DIR}/")
print(f"  Gráficos em          : {GRAFICOS_DIR}/")
