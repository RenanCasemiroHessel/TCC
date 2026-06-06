
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
plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,"figure.titlesize":17})
import seaborn as sns

from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED          = 42
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_MAX    = 250.0   
SAMPLE_FRAC   = 1.0
SAVE_DIR      = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR  = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  CNN Baseline para Estimativa de Peso Bovino")
print(f"  Dispositivo : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"  Épocas      : {NUM_EPOCHS}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  SAMPLE_FRAC : {SAMPLE_FRAC}")
print(f"  Peso máx    : <{WEIGHT_MAX:.0f} kg (somente leves)")

print("Carregando dataset via kagglehub ...")
import kagglehub

dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")

base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

def extrair_peso(filename, caminho):
    nome = filename.replace("___fuse.png", "").replace(".jpg", "")
    partes = nome.split("_")
    try:
        indice = 3 if "B4" in caminho or "b4" in caminho else 2
        val = float(partes[indice])
        if 50 <= val < WEIGHT_MAX:
            return val
    except:
        pass
    return None

records = []
for root, dirs, files in os.walk(base_dir):
    if os.path.basename(root) == "images":
        pasta_imgs = root
        pasta_masks = os.path.join(os.path.dirname(root), "annotations")

        if not os.path.exists(pasta_masks):
            continue

        rel_path = os.path.relpath(root, base_dir)
        partes_caminho = rel_path.split(os.sep)
        lote = partes_caminho[0]

        if lote == "B3":
            continue

        for img_file in sorted(os.listdir(pasta_imgs)):
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
df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

print(f"      Shape do dataset estruturado: {df.shape}")
print(df.head(3))

img_col    = "image_path"
weight_col = "weight"

print(f"\n      Coluna imagem : '{img_col}'")

df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
df = df.dropna(subset=[img_col, weight_col]).copy()

print(f"\n      Registros após filtro (sem B3, leves <{WEIGHT_MAX:.0f}kg, sem duplicatas): {len(df)}")

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

if len(sample_masks) > 0:
    print(f"Máscaras encontradas: {masks_found}/100")
else:
    print("Aviso: nenhuma máscara encontrada na amostra.")

print("\nDividindo dados (80% treino / 20% validação) ...")

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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
            image_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(image_pil)
            
            dirname, filename = os.path.split(img_path)
            parent_dir, last_dir = os.path.split(dirname)
            
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
                
                if mask_np.shape[-1] == 3:
                    alpha = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
                    mask_np = np.concatenate([mask_np, alpha], axis=-1)
                
                bovino_mask = np.all(mask_np == cor_alvo, axis=-1)
                
                masked_img = img_np.copy()
                masked_img[~bovino_mask] = 0
                image_pil = Image.fromarray(masked_img)
            else:
                image_pil = Image.fromarray(img_np)
                
        except Exception:
            image_pil = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            
        if self.transform:
            image_pil = self.transform(image_pil)
            
        return image_pil, torch.tensor(weight, dtype=torch.float32)

train_dataset = CattleWeightDataset(train_df, img_col, weight_col, train_transform)
val_dataset   = CattleWeightDataset(val_df,   img_col, weight_col, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=torch.cuda.is_available())

print("\nConstruindo modelo (ResNet18 pré-treinado) ...")

train_weights = train_df[weight_col].values
Y_MIN = float(train_weights.min())
Y_MAX = float(train_weights.max())

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)

for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

model = model.to(DEVICE)

huber_delta = 50.0 / (Y_MAX - Y_MIN)  # delta normalizado para escala [0,1]
criterion = nn.HuberLoss(delta=huber_delta)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)
FINETUNE_EPOCH = 3   # descongelar mais cedo para mais épocas de fine-tuning

print(f"\nIniciando treino por {NUM_EPOCHS} épocas ...")
print(f"      Estratégia: Épocas 1-{FINETUNE_EPOCH-1} → só cabeça | Épocas {FINETUNE_EPOCH}+ → backbone descongelado\n")

from utils_tcc import safe_path, salvar_json
MODEL_PATH = safe_path(os.path.join(SAVE_DIR, "resnet18_gado_segmentado.pth"))
print(f"      Modelo será salvo em: {os.path.basename(MODEL_PATH)}")

history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
best_epoch    = 0

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    if epoch == FINETUNE_EPOCH:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE * 0.1)  # LR menor para fine-tune
        print(f"  [Época {epoch}] Backbone DESCONGELADO — fine-tuning completo com LR={LEARNING_RATE * 0.1:.6f}\n")

    model.train()
    running_loss = 0.0
    for imgs, weights in train_loader:
        imgs    = imgs.to(DEVICE)
        w_norm  = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        preds   = model(imgs)          # saída em [0,1] por causa do Sigmoid
        loss    = criterion(preds, w_norm)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_dataset)

    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for imgs, weights in val_loader:
            imgs   = imgs.to(DEVICE)
            w_norm = ((weights - Y_MIN) / (Y_MAX - Y_MIN)).to(DEVICE).unsqueeze(1)
            preds  = model(imgs)
            val_running += criterion(preds, w_norm).item() * imgs.size(0)
    val_loss = val_running / len(val_dataset)

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

print("\nCalculando métricas no conjunto de validação ...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"      Usando pesos da época {best_epoch} (melhor val loss: {best_val_loss:.6f})")

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, weights in val_loader:
        imgs  = imgs.to(DEVICE)
        preds = model(imgs).squeeze(1).cpu().numpy()
        preds_kg = preds * (Y_MAX - Y_MIN) + Y_MIN
        all_preds.extend(preds_kg.tolist())
        all_true.extend(weights.numpy().tolist())

y_true = np.array(all_true)
y_pred = np.array(all_preds)
errors = y_pred - y_true

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

pearson_r = np.corrcoef(y_true, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — CNN Baseline (ResNet18)
{'='*60}

    print("\nResultados:")
    print(f"MAE  (Erro Absoluto Médio)  : {mae:>8.2f} kg")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg")
    print(f"R²  (Coef. de Determinação)  : {r2:>8.4f}")
    print(f"Pearson r  : {pearson_r:>8.4f}")
""")

print("  Gerando gráficos ...\n")
DPI = 150

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
path_loss = os.path.join(GRAFICOS_DIR, "cnn_curva_loss.png")
fig.savefig(path_loss, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_curva_loss.png")

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
path_scatter = os.path.join(GRAFICOS_DIR, "cnn_grafico_dispersao_r2.png")
fig.savefig(path_scatter, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_grafico_dispersao_r2.png")

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
path_hist = os.path.join(GRAFICOS_DIR, "cnn_histograma_erros.png")
fig.savefig(path_hist, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_histograma_erros.png")

fig, ax = plt.subplots(figsize=(7, 6))
ax.boxplot(errors, vert=True, patch_artist=True,
           boxprops=dict(facecolor="#7C3AED", alpha=0.6),
           medianprops=dict(color="black", linewidth=2),
           flierprops=dict(marker="o", markerfacecolor="grey", markersize=3, alpha=0.5))
ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Erro = 0")
ax.set_title("CNN — Distribuição dos Erros de Predição", fontsize=14,
             fontweight="bold", pad=12)
ax.set_ylabel("Erro (Predito − Real) [kg]", fontsize=12)
ax.set_xticks([1]); ax.set_xticklabels(["CNN (ResNet18)"])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
path_box = os.path.join(GRAFICOS_DIR, "cnn_boxplot_erros.png")
fig.savefig(path_box, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_boxplot_erros.png")

pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_true, y_pred, alpha=0.4, s=18, color="#7C3AED", edgecolors="none", label="Amostras")
coef_corr = np.polyfit(y_true, y_pred, 1)
x_line = np.linspace(y_true.min(), y_true.max(), 100)
ax.plot(x_line, np.polyval(coef_corr, x_line), color="#E74C3C", linewidth=2,
        label=f"Tendência (r = {pearson_r:.4f})")
ax.plot(x_line, x_line, "k--", linewidth=1.2, alpha=0.5, label="y = x (perfeito)")
ax.set_title(f"CNN — Correlação Linear\nPearson r = {pearson_r:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)", fontsize=12)
ax.set_ylabel("Peso Previsto (kg)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path_corr = os.path.join(GRAFICOS_DIR, "cnn_correlacao_linear.png")
fig.savefig(path_corr, dpi=DPI)
plt.close(fig)
print(f"  [OK] cnn_correlacao_linear.png")

print(f"  Treino concluído. Gráficos em:")
print(f"  {GRAFICOS_DIR}")

RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
npy_path = safe_path(os.path.join(SAVE_DIR, "predictions_cnn.npy"))
np.save(npy_path, np.array([y_true, y_pred]))
salvar_json(RESULTS_PATH, "CNN (ResNet18)", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
    "train_history": history["train_loss"], "val_history": history["val_loss"],
})
print(f"  Resultados salvos em: resultados_modelos.json\n")
print(f"  Modelo salvo em: {os.path.basename(MODEL_PATH)}  (época {best_epoch})")
print(f"  (Use predizer_imagem.py para inferência em novas imagens)\n")

