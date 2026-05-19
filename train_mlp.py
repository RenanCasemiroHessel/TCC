"""
TCC FEI — MLP (Multi-Layer Perceptron) para Estimativa de Peso Bovino
Pipeline: Imagens segmentadas (fundo preto) achatadas em vetor de pixels (como MNIST)
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

from PIL import Image
import kagglehub

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
IMG_SIZE   = 64           # 64x64 pixels (como MNIST 28x28, mas maior)
WEIGHT_MIN = 50.0
WEIGHT_MAX = 1000.0

np.random.seed(SEED)

print(f"\n{'='*60}")
print(f"  TCC FEI — MLP para Estimativa de Peso Bovino")
print(f"{'='*60}")
print(f"  Entrada  : Imagens segmentadas {IMG_SIZE}x{IMG_SIZE} (achatadas)")
print(f"  Modelo   : MLPRegressor (sklearn)")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAR DATASET (mesma estrutura do treinar_cnn.py)
# ─────────────────────────────────────────────────────────────────────────────
print("[1/5] Carregando dataset via kagglehub ...")
dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

pastas_lote = {
    "B2": os.path.join(base_dir, "B2", "Side"),
    "B3": os.path.join(base_dir, "B3"),
    "B4": os.path.join(base_dir, "B4", "Side"),
}

def extrair_peso(filename, caminho):
    nome = filename.replace(".jpg", "")
    partes = nome.split("_")
    try:
        indice = 3 if "B4" in caminho or "b4" in caminho else 2
        val = float(partes[indice])
        if WEIGHT_MIN <= val <= WEIGHT_MAX:
            return val
    except Exception:
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
        
        # B2 usa máscara verde ([0, 255, 0, 255]), os demais (B3, B4) usam magenta
        cor_alvo = np.array([0, 255, 0, 255]) if "B2" in lote else np.array([255, 30, 249, 255])
        
        for img_file in os.listdir(pasta_imgs):
            if not img_file.endswith(".jpg"):
                continue
                
            peso = extrair_peso(img_file, rel_path)
            if peso is None:
                continue
                
            img_path  = os.path.join(pasta_imgs, img_file)
            mask_path = os.path.join(pasta_masks, img_file + "___fuse.png")
            if not os.path.exists(mask_path):
                continue
                
            records.append({
                "image_path": img_path,
                "mask_path":  mask_path,
                "weight":     peso,
                "lote":       lote,
                "cor_alvo":   cor_alvo.tolist(),
            })

df = pd.DataFrame(records)
print(f"      Total de amostras localizadas recursivamente: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRAIR PIXELS DAS IMAGENS SEGMENTADAS (com cache de arquivo para ultra-velocidade!)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Extraindo pixels das imagens segmentadas...")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_X = os.path.join(SAVE_DIR, f"mlp_cache_X_{IMG_SIZE}.npy")
CACHE_Y = os.path.join(SAVE_DIR, f"mlp_cache_y_{IMG_SIZE}.npy")

if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    print("      [CACHE] Carregando pixels pré-processados salvos anteriormente...")
    X = np.load(CACHE_X)
    y = np.load(CACHE_Y)
else:
    print("      [Aviso] Cache não encontrada. Processando imagens pela primeira vez (isso pode demorar)...")
    X_pixels = []
    y_labels = []

    for i, row in df.iterrows():
        try:
            img_np  = np.array(Image.open(row["image_path"]).convert("RGB"))
            mask_np = np.array(Image.open(row["mask_path"]))

            if mask_np.shape[-1] == 3:
                alpha   = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
                mask_np = np.concatenate([mask_np, alpha], axis=-1)

            cor_alvo    = np.array(row["cor_alvo"])
            bovino_mask = np.all(mask_np == cor_alvo, axis=-1)

            masked_img           = img_np.copy()
            masked_img[~bovino_mask] = 0

            # Redimensionar para IMG_SIZE x IMG_SIZE (como MNIST → 28x28)
            img_small = np.array(
                Image.fromarray(masked_img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            )
            # Achatar (flatten) — igual ao MNIST mas com 64*64*3 = 12288 features
            X_pixels.append(img_small.flatten().astype(np.float32) / 255.0)
            y_labels.append(row["weight"])

        except Exception:
            continue

    X = np.array(X_pixels)
    y = np.array(y_labels)
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    print("      [CACHE] Pixels salvos para execuções instantâneas futuras!")

print(f"      Shape X (pixels): {X.shape}  |  Shape y: {y.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPLIT 80/20 & PCA (Redução de Dimensionalidade)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Dividindo dados (80% treino / 20% validação) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"      Treino: {len(X_train)} | Validação: {len(X_test)}")

# Normalização (Z-score, igual ao MNIST normalizado)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# PCA para evitar Overfitting e acelerar o treino em 100x!
print("      Aplicando PCA para extração de 128 componentes principais mais importantes...")
from sklearn.decomposition import PCA
pca = PCA(n_components=128, random_state=SEED)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)
print(f"      Novo shape após PCA: X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TREINO DO MLP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Treinando MLP ...")
print("      Arquitetura: 128 (PCA) → 256 → 128 → 64 → 1 (regressão)")

mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=SEED,
    verbose=False,
)
mlp.fit(X_train, y_train)
print(f"      Convergiu em {mlp.n_iter_} iterações")

y_pred = mlp.predict(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# Classificação por faixas biológicas
class_names = ["Leve (<250kg)", "Médio (250-450kg)", "Pesado (>450kg)"]
def classify(w):
    if w < 250:   return 0
    if w <= 450:  return 1
    return 2

y_true_cat = [classify(v) for v in y_test]
y_pred_cat = [classify(v) for v in y_pred]

acc  = accuracy_score(y_true_cat, y_pred_cat)
prec = precision_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)
rec  = recall_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)
f1   = f1_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)

print(f"""
{'='*60}
  RELATÓRIO FINAL — MLP (pixels segmentados {IMG_SIZE}x{IMG_SIZE})
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
# 6. GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
print("[5/5] Gerando gráficos ...\n")
DPI      = 150
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
errors   = y_pred - y_test

# 6.1 Curva de Loss (loss_curve_ do MLPRegressor)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(mlp.loss_curve_, color="#4A90D9", linewidth=2, label="Loss Treino")
if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
    ax.plot([-v for v in mlp.validation_scores_], color="#E74C3C",
            linewidth=2, linestyle="--", label="Loss Validação")
ax.set_title("MLP — Curva de Loss por Iteração", fontsize=14, fontweight="bold")
ax.set_xlabel("Iteração")
ax.set_ylabel("Loss (MSE)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mlp_curva_loss.png"), dpi=DPI)
plt.close()
print("    ✓ mlp_curva_loss.png")

# 6.2 Dispersão R²
fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#4A90D9", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"MLP — Dispersão Real vs Previsto  (R²={r2:.4f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mlp_grafico_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    ✓ mlp_grafico_dispersao_r2.png")

# 6.3 Histograma de Erros
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#4A90D9", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("MLP — Histograma de Erros (Previsto − Real)", fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mlp_histograma_erros.png"), dpi=DPI)
plt.close()
print("    ✓ mlp_histograma_erros.png")

# 6.4 Boxplot/Violino de Erros por Faixa
df_err = pd.DataFrame({
    "Erro": errors,
    "Faixa": [class_names[c] for c in y_true_cat]
})
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.violinplot(data=df_err, x="Faixa", y="Erro", palette="Set2", ax=axes[0])
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("MLP — Violino de Erros por Faixa")
axes[0].set_ylabel("Erro (kg)")
sns.boxplot(data=df_err, x="Faixa", y="Erro", palette="Set2", ax=axes[1])
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("MLP — Boxplot de Erros por Faixa")
axes[1].set_ylabel("Erro (kg)")
plt.suptitle("MLP — Distribuição dos Erros por Faixa Biológica", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mlp_boxplot_violino_erros.png"), dpi=DPI)
plt.close()
print("    ✓ mlp_boxplot_violino_erros.png")

print(f"\n{'='*60}")
print(f"  Gráficos gerados em: {SAVE_DIR}")
print(f"{'='*60}\n")

import json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
existing = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        existing = json.load(f)
existing["MLP (pixels)"] = {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
    "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
    "train_history": mlp.loss_curve_, "val_history": [],
}
with open(RESULTS_PATH, "w") as f:
    json.dump(existing, f, indent=2)
print("  Resultados salvos em: resultados_modelos.json\n")
