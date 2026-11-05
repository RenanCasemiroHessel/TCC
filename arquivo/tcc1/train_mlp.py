"""
MLP (Multi-Layer Perceptron) para Estimativa de Peso Bovino
Pipeline: Imagens segmentadas (fundo preto) achatadas em vetor de pixels (como MNIST)
Dataset: apenas leves (<250 kg), sem B3, sem duplicatas
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
import seaborn as sns

from PIL import Image
import kagglehub

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED       = 42
IMG_SIZE   = 64           # 64x64 pixels (como MNIST 28x28, mas maior)
WEIGHT_MAX = 250.0        # Dataset filtrado: só leves

np.random.seed(SEED)

print(f"  MLP para Estimativa de Peso Bovino")
print(f"  Entrada  : Imagens segmentadas {IMG_SIZE}x{IMG_SIZE} (achatadas)")
print(f"  Modelo   : MLPRegressor (sklearn)")
print(f"  Dataset  : apenas leves (<{WEIGHT_MAX:.0f}kg), sem B3")

print("Carregando dataset via kagglehub ...")
dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

def extrair_peso(filename, caminho):
    nome = filename.replace(".jpg", "")
    partes = nome.split("_")
    try:
        indice = 3 if "B4" in caminho or "b4" in caminho else 2
        val = float(partes[indice])
        if 50 <= val < WEIGHT_MAX:
            return val
    except Exception:
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

        cor_alvo = np.array([0, 255, 0, 255]) if "B2" in lote else np.array([255, 30, 249, 255])

        for img_file in sorted(os.listdir(pasta_imgs)):
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

df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

print("\nExtraindo pixels das imagens segmentadas...")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)
CACHE_X = os.path.join(SAVE_DIR, f"mlp_cache_X_{IMG_SIZE}_leves.npy")
CACHE_Y = os.path.join(SAVE_DIR, f"mlp_cache_y_{IMG_SIZE}_leves.npy")

if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    print("      [CACHE] Carregando pixels pré-processados salvos anteriormente...")
    X = np.load(CACHE_X)
    y = np.load(CACHE_Y)
else:
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

            img_small = np.array(
                Image.fromarray(masked_img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            )
            X_pixels.append(img_small.flatten().astype(np.float32) / 255.0)
            y_labels.append(row["weight"])

        except Exception:
            continue

    X = np.array(X_pixels)
    y = np.array(y_labels)
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    print("      [CACHE] Pixels salvos para execuções instantâneas futuras!")

print("\nDividindo dados (80% treino / 20% validação) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=128, random_state=SEED)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

print("\nTreinando MLP ...")

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

y_pred = mlp.predict(X_test)
WEIGHT_MIN = 50.0
y_pred = np.clip(y_pred, WEIGHT_MIN, WEIGHT_MAX)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

pearson_r = np.corrcoef(y_test, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — MLP (pixels segmentados {IMG_SIZE}x{IMG_SIZE})
{'='*60}

    print("\nResultados:")
    print(f"MAE  (Erro Absoluto Médio)  : {mae:>8.2f} kg")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg")
    print(f"R²  (Coef. de Determinação)  : {r2:>8.4f}")
    print(f"Pearson r  : {pearson_r:>8.4f}")
""")

print("Gerando gráficos ...\n")
DPI    = 150
errors = y_pred - y_test

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
plt.savefig(os.path.join(GRAFICOS_DIR, "mlp_curva_loss.png"), dpi=DPI)
plt.close()
print("    [OK] mlp_curva_loss.png")

fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#4A90D9", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"MLP — Dispersão Real vs Previsto  (R²={r2:.4f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "mlp_grafico_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    [OK] mlp_grafico_dispersao_r2.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#4A90D9", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("MLP — Histograma de Erros (Previsto − Real)", fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "mlp_histograma_erros.png"), dpi=DPI)
plt.close()
print("    [OK] mlp_histograma_erros.png")

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred, alpha=0.5, color="#4A90D9", edgecolors="white", linewidths=0.3, s=40, label="Amostras")
coef_corr = np.polyfit(y_test, y_pred, 1)
x_line = np.linspace(y_test.min(), y_test.max(), 100)
ax.plot(x_line, np.polyval(coef_corr, x_line), color="#E74C3C", linewidth=2,
        label=f"Tendência (r = {pearson_r:.4f})")
ax.plot(x_line, x_line, "k--", linewidth=1.2, alpha=0.5, label="y = x (perfeito)")
ax.set_title(f"MLP — Correlação Linear\nPearson r = {pearson_r:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "mlp_correlacao_linear.png"), dpi=DPI)
plt.close()
print("    [OK] mlp_correlacao_linear.png")

print(f"  Gráficos gerados em: {GRAFICOS_DIR}")

from utils_tcc import safe_path, salvar_json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
npy_path = safe_path(os.path.join(SAVE_DIR, "predictions_mlp.npy"))
np.save(npy_path, np.array([y_test, y_pred]))
salvar_json(RESULTS_PATH, "MLP (pixels)", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
    "train_history": mlp.loss_curve_, "val_history": [],
})
print("  Resultados salvos em: resultados_modelos.json\n")
