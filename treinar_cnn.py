"""
TCC FEI — CNN Baseline (sem segmentação) para Estimativa de Peso Bovino
Arquitetura: ResNet18 pré-treinada com regressão de saída única
"""

import os
import re
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED         = 42
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 15
LEARNING_RATE = 1e-4
WEIGHT_MIN   = 50.0
WEIGHT_MAX   = 1000.0

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  TCC FEI — CNN Baseline para Estimativa de Peso Bovino")
print(f"{'='*60}")
print(f"  Dispositivo : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"  Épocas      : {NUM_EPOCHS}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD E CURADORIA DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────
print("[1/5] Carregando dataset via kagglehub ...")
import kagglehub

# Obter o caminho da pasta raiz do dataset pelo kagglehub
dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")

# Resolver o caminho completo do Pixel
base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

pastas_lote = {
    "B2": os.path.join(base_dir, "B2", "Side"),
    "B3": os.path.join(base_dir, "B3"),
    "B4": os.path.join(base_dir, "B4", "Side"),
}

def extrair_peso(filename, caminho):
    nome = filename.replace("___fuse.png", "").replace(".jpg", "")
    partes = nome.split("_")
    try:
        indice = 3 if "B4" in caminho or "b4" in caminho else 2
        val = float(partes[indice])
        if 50 <= val <= 1000:
            return val
    except:
        pass
    return None

records = []
print("      Escaneando todos os subdiretórios recursivamente...")
for root, dirs, files in os.walk(base_dir):
    if os.path.basename(root) == "images":
        pasta_imgs = root
        pasta_masks = os.path.join(os.path.dirname(root), "annotations")
        
        if not os.path.exists(pasta_masks):
            continue
            
        rel_path = os.path.relpath(root, base_dir)
        partes_caminho = rel_path.split(os.sep)
        lote = partes_caminho[0]
        
        for img_file in os.listdir(pasta_imgs):
            if not img_file.endswith(".jpg"):
                continue
                
            peso = extrair_peso(img_file, rel_path)
            if peso is None:
                continue
                
            img_path = os.path.join(pasta_imgs, img_file)
            mask_file = img_file + "___fuse.png"
            mask_path = os.path.join(pasta_masks, mask_file)
            
            if not os.path.exists(mask_path):
                continue
                
            records.append({
                "image_path": img_path,
                "weight": peso,
                "lote": lote
            })

df = pd.DataFrame(records)

print(f"      Shape do dataset estruturado: {df.shape}")
print(df.head(3))

img_col    = "image_path"
weight_col = "weight"

print(f"\n      Coluna imagem : '{img_col}'")
print(f"      Coluna peso   : '{weight_col}'")

# Garantir tipos corretos
df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
df = df.dropna(subset=[img_col, weight_col]).copy()

# ── Filtro de Domínio ────────────────────────────────────────────────────────
before = len(df)
df = df[(df[weight_col] >= WEIGHT_MIN) & (df[weight_col] <= WEIGHT_MAX)].copy()
df = df.reset_index(drop=True)
print(f"\n      Registos após filtro de domínio [{WEIGHT_MIN}-{WEIGHT_MAX}kg]: {len(df)} (removidos: {before - len(df)})")

# ── Verificação Física das Máscaras ──────────────────────────────────────────
print("\n[VERIFICAÇÃO DE MÁSCARAS]")
masks_found = 0
sample_masks = []
for idx, row in df.head(100).iterrows():
    img_path = str(row[img_col])
    dirname, filename = os.path.split(img_path)
    parent_dir, last_dir = os.path.split(dirname)
    if last_dir == "images":
        mask_dir = os.path.join(parent_dir, "annotations")
        mask_file = filename + "___fuse.png"
        mask_path = os.path.join(mask_dir, mask_file)
        if os.path.exists(mask_path):
            masks_found += 1
            if len(sample_masks) < 3:
                sample_masks.append((img_path, mask_path))

print(f"      Máscaras encontradas na amostra inicial (100 linhas): {masks_found}")
if len(sample_masks) > 0:
    print("      [SUCESSO] Máscaras de segmentação localizadas fisicamente!")
    print("      Estrutura de caminhos de exemplo:")
    for img_p, m_p in sample_masks:
        print(f"        - Imagem:  {img_p}")
        print(f"        - Máscara: {m_p}")
else:
    print("      [AVISO] Nenhuma máscara foi encontrada na amostra (ainda baixando/extraindo).")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT 80/20
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Dividindo dados (80% treino / 20% validação) ...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
print(f"      Treino: {len(train_df)} | Validação: {len(val_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET E DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CattleWeightDataset(Dataset):
    def __init__(self, dataframe, img_col, weight_col, transform=None):
        self.df         = dataframe.reset_index(drop=True)
        self.img_col    = img_col
        self.weight_col = weight_col
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = str(self.df.loc[idx, self.img_col])
        weight   = float(self.df.loc[idx, self.weight_col])
        
        try:
            # 1. Carregar Imagem Original
            image_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(image_pil)
            
            # 2. Localizar Máscara
            dirname, filename = os.path.split(img_path)
            parent_dir, last_dir = os.path.split(dirname)
            
            # Lógica de cor padrão por lote
            # Lote B2: Verde [0, 255, 0, 255]
            # Lotes B3 e B4: Rosa/Magenta [255, 30, 249, 255]
            if "B2" in img_path:
                cor_alvo = np.array([0, 255, 0, 255])
            else:
                cor_alvo = np.array([255, 30, 249, 255])
                
            mask_path = None
            if last_dir == "images":
                mask_dir = os.path.join(parent_dir, "annotations")
                mask_file = filename + "___fuse.png"
                mask_path = os.path.join(mask_dir, mask_file)
            
            if mask_path and os.path.exists(mask_path):
                mask_pil = Image.open(mask_path)
                mask_np = np.array(mask_pil)
                
                # Ajustar canais da máscara caso não tenha canal Alpha
                if mask_np.shape[-1] == 3:
                    alpha = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
                    mask_np = np.concatenate([mask_np, alpha], axis=-1)
                
                # Criar máscara binária do boi
                bovino_mask = np.all(mask_np == cor_alvo, axis=-1)
                
                # Aplicar fundo preto (isolar boi)
                masked_img = img_np.copy()
                masked_img[~bovino_mask] = 0
                image_pil = Image.fromarray(masked_img)
            else:
                # Fallback para usar a imagem original
                image_pil = Image.fromarray(img_np)
                
        except Exception:
            # Fallback em caso de erro na leitura/processamento
            image_pil = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            
        if self.transform:
            image_pil = self.transform(image_pil)
            
        return image_pil, torch.tensor(weight, dtype=torch.float32)


train_dataset = CattleWeightDataset(train_df, img_col, weight_col, train_transform)
val_dataset   = CattleWeightDataset(val_df,   img_col, weight_col, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=torch.cuda.is_available())

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELO — ResNet18 com cabeça de regressão
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Construindo modelo (ResNet18 pré-treinado) ...")

# ── Normalização do alvo (crítico para convergência da regressão) ─────────────
# Normaliza pesos para [0,1] durante treino e escala de volta nas métricas
train_weights = train_df[weight_col].values
Y_MIN = float(train_weights.min())
Y_MAX = float(train_weights.max())
print(f"      Normalização do alvo: [{Y_MIN:.0f}, {Y_MAX:.0f}] kg → [0, 1]")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features

# Cabeça de regressão com sigmoid para garantir saída em [0,1]
model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)

# Fase 1: Congelar backbone, treinar só a cabeça (primeiras épocas)
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

model = model.to(DEVICE)
print(f"      Parâmetros treináveis (fase 1 - só cabeça): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = nn.MSELoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True
)
FINETUNE_EPOCH = 5   # a partir desta época, descongela o backbone

# ─────────────────────────────────────────────────────────────────────────────
# 5. LOOP DE TREINO
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Iniciando treino por {NUM_EPOCHS} épocas ...")
print(f"      Estratégia: Épocas 1-{FINETUNE_EPOCH-1} → só cabeça | Épocas {FINETUNE_EPOCH}+ → backbone descongelado\n")

history = {"train_loss": [], "val_loss": []}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    # ── Fase 2: Descongelar backbone a partir de FINETUNE_EPOCH ──
    if epoch == FINETUNE_EPOCH:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE * 0.1)  # LR menor para fine-tune
        print(f"  [Época {epoch}] Backbone DESCONGELADO — fine-tuning completo com LR={LEARNING_RATE * 0.1:.6f}\n")

    # ── Treino (alvos normalizados para [0,1]) ──
    model.train()
    running_loss = 0.0
    for imgs, weights in train_loader:
        imgs    = imgs.to(DEVICE)
        # Normaliza peso para [0, 1]
        w_norm  = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        preds   = model(imgs)          # saída em [0,1] por causa do Sigmoid
        loss    = criterion(preds, w_norm)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_dataset)

    # ── Validação ──
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for imgs, weights in val_loader:
            imgs   = imgs.to(DEVICE)
            w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
            preds  = model(imgs)
            val_running += criterion(preds, w_norm).item() * imgs.size(0)
    val_loss = val_running / len(val_dataset)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    elapsed = time.time() - t0
    print(f"  Época {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"Tempo: {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MÉTRICAS DE AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Calculando métricas no conjunto de validação ...")

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, weights in val_loader:
        imgs  = imgs.to(DEVICE)
        preds = model(imgs).squeeze(1).cpu().numpy()
        # Desnormalizar: voltar para kg reais
        preds_kg = preds * (Y_MAX - Y_MIN) + Y_MIN
        all_preds.extend(preds_kg.tolist())
        all_true.extend(weights.numpy().tolist())

y_true = np.array(all_true)
y_pred = np.array(all_preds)
errors = y_pred - y_true

# ── Métricas de Regressão ────────────────────────────────────────────────────
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

# ── Classificação por Faixas Biológicas ─────────────────────────────────────
def discretize(weights):
    cats = np.where(weights < 250, 0, np.where(weights <= 450, 1, 2))
    return cats

y_true_cat = discretize(y_true)
y_pred_cat = discretize(y_pred)

class_names = ["Leve (<250kg)", "Médio (250-450kg)", "Pesado (>450kg)"]
acc  = accuracy_score(y_true_cat, y_pred_cat)
prec = precision_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)
rec  = recall_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)
f1   = f1_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)

# ── Relatório Final ──────────────────────────────────────────────────────────
print(f"""
{'='*60}
  RELATÓRIO FINAL — CNN Baseline (ResNet18)
{'='*60}

  ┌─ MÉTRICAS DE REGRESSÃO ─────────────────────────────────┐
  │  MAE  (Erro Absoluto Médio)    : {mae:>8.2f} kg           │
  │  RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg           │
  │  R²   (Coef. de Determinação)  : {r2:>8.4f}               │
  └──────────────────────────────────────────────────────────┘

  ┌─ MÉTRICAS DE CLASSIFICAÇÃO (3 Faixas Biológicas) ───────┐
  │  Acurácia : {acc*100:>6.2f}%                                   │
  │  Precisão : {prec*100:>6.2f}% (weighted)                        │
  │  Recall   : {rec*100:>6.2f}% (weighted)                        │
  │  F1-Score : {f1*100:>6.2f}% (weighted)                         │
  └──────────────────────────────────────────────────────────┘
""")

print("  Relatório de Classificação Detalhado:")
print(classification_report(y_true_cat, y_pred_cat,
                             labels=[0, 1, 2], target_names=class_names, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# 7. GERAÇÃO DOS GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
print("  Gerando gráficos ...\n")
DPI = 150
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 7.1 Curva de Loss ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
epochs_range = range(1, NUM_EPOCHS + 1)
ax.plot(epochs_range, history["train_loss"], "o-", color="#2563EB", linewidth=2.5,
        markersize=7, label="Treino MSE")
ax.plot(epochs_range, history["val_loss"],   "s--", color="#DC2626", linewidth=2.5,
        markersize=7, label="Validação MSE")
ax.set_title("Curva de Loss — Treino vs Validação", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Época", fontsize=12)
ax.set_ylabel("MSE Loss (kg²)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.35)
ax.set_xticks(list(epochs_range))
plt.tight_layout()
path_loss = os.path.join(SAVE_DIR, "cnn_curva_loss.png")
fig.savefig(path_loss, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_curva_loss.png")

# ─── 7.2 Scatter Real × Predito ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, alpha=0.45, s=18, color="#7C3AED", edgecolors="none", label="Amostras")
lims = [min(y_true.min(), y_pred.min()) - 20, max(y_true.max(), y_pred.max()) + 20]
ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x (perfeito)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"Real × Predito  |  R² = {r2:.4f}", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Peso Real (kg)", fontsize=12)
ax.set_ylabel("Peso Predito (kg)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path_scatter = os.path.join(SAVE_DIR, "cnn_grafico_dispersao_r2.png")
fig.savefig(path_scatter, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_grafico_dispersao_r2.png")

# ─── 7.3 Histograma de Erros / Resíduos ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=50, color="#059669", edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(0, color="black", linewidth=1.8, linestyle="--", label="Erro = 0")
ax.axvline(errors.mean(), color="#DC2626", linewidth=1.8, linestyle="-",
           label=f"Média: {errors.mean():.1f} kg")
ax.set_title("Distribuição dos Resíduos (Erros de Predição)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Erro (Predito − Real) [kg]", fontsize=12)
ax.set_ylabel("Frequência", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
path_hist = os.path.join(SAVE_DIR, "cnn_histograma_erros.png")
fig.savefig(path_hist, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_histograma_erros.png")

# ─── 7.4 Violino + Boxplot do Erro Absoluto por Faixa ────────────────────────
abs_errors = np.abs(errors)
faixa_labels = np.array(class_names)
faixas_true  = faixa_labels[y_true_cat]

plot_df = pd.DataFrame({"Faixa": faixas_true, "Erro Absoluto (kg)": abs_errors})
order = class_names

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=plot_df, x="Faixa", y="Erro Absoluto (kg)", order=order,
               inner=None, palette=["#3B82F6", "#F59E0B", "#EF4444"],
               linewidth=1.2, ax=ax, alpha=0.7)
sns.boxplot(data=plot_df, x="Faixa", y="Erro Absoluto (kg)", order=order,
            width=0.15, flierprops=dict(marker="o", markerfacecolor="k", markersize=3),
            boxprops=dict(facecolor="white", zorder=2), linewidth=1.5, ax=ax)
ax.set_title("Erro Absoluto por Faixa de Peso (Violino + Boxplot)", fontsize=14,
             fontweight="bold", pad=12)
ax.set_xlabel("Faixa de Peso Biológica", fontsize=12)
ax.set_ylabel("Erro Absoluto (kg)", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
path_violin = os.path.join(SAVE_DIR, "cnn_boxplot_violino_erros.png")
fig.savefig(path_violin, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_boxplot_violino_erros.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Treino concluído. Ficheiros gerados em:")
print(f"  {SAVE_DIR}")
print(f"{'='*60}\n")

# ── Salvar resultados para comparação entre modelos ──────────────────────────
import json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
existing = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        existing = json.load(f)

existing["CNN (ResNet18)"] = {
    "mae":          float(mae),
    "rmse":         float(rmse),
    "r2":           float(r2),
    "accuracy":     float(acc),
    "precision":    float(prec),
    "recall":       float(rec),
    "f1":           float(f1),
    "train_history": history["train_loss"],
    "val_history":   history["val_loss"],
}
with open(RESULTS_PATH, "w") as f:
    json.dump(existing, f, indent=2)
print(f"  Resultados salvos em: resultados_modelos.json\n")

