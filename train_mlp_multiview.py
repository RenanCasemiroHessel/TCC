"""
MLP Multi-View
Carrega pixels de vista lateral + traseira do mesmo animal, concatena e treina MLP.
"""

import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,"figure.titlesize":17})

from PIL import Image
import kagglehub

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED       = 42
IMG_SIZE   = 64
WEIGHT_MAX = 250.0
np.random.seed(SEED)

SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

print(f"  MLP Multi-View (side + rear pixels)")

print("Escaneando dataset e montando pares side+rear ...")

dataset_path = kagglehub.dataset_download("sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
base_dir = os.path.join(dataset_path, "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")

def parse_filename(filename, lote):
    name  = filename.replace(".jpg", "")
    parts = name.split("_")
    try:
        if lote == "B2":
            animal_id = parts[0]
            weight    = float(parts[2])
        else:
            animal_id = f"{parts[0]}_{parts[1]}"
            weight    = float(parts[3])
        if 50 <= weight < WEIGHT_MAX:
            return animal_id, weight
    except Exception:
        pass
    return None

sides = {}
rears = {}

for root, dirs, files in os.walk(base_dir):
    if os.path.basename(root) != "images":
        continue
    rel         = os.path.relpath(root, base_dir)
    parts_path  = rel.split(os.sep)
    lote        = parts_path[0]
    if lote == "B3":
        continue
    vista_dir   = parts_path[1] if len(parts_path) > 1 else ""
    is_side     = "Side" in vista_dir
    is_rear     = "Back" in vista_dir or "Rear" in vista_dir
    if not (is_side or is_rear):
        continue

    pasta_masks = os.path.join(os.path.dirname(root), "annotations")
    if not os.path.exists(pasta_masks):
        continue

    cor = np.array([0, 255, 0, 255]) if lote == "B2" else np.array([255, 30, 249, 255])

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
        key   = f"{lote}_{animal_id}_{weight}"
        entry = {"img": img_path, "mask": mask_path, "weight": weight, "cor": cor}
        if is_side and key not in sides:
            sides[key] = entry
        elif is_rear and key not in rears:
            rears[key] = entry

pairs = [(k, sides[k], rears[k]) for k in sides if k in rears]

print("\nExtraindo pixels side+rear (cache) ...")
CACHE_X = os.path.join(SAVE_DIR, f"mlp_multiview_cache_X_{IMG_SIZE}.npy")
CACHE_Y = os.path.join(SAVE_DIR, f"mlp_multiview_cache_y_{IMG_SIZE}.npy")

def segmentar(img_path, mask_path, cor):
    img_np  = np.array(Image.open(img_path).convert("RGB"))
    mask_np = np.array(Image.open(mask_path))
    if mask_np.shape[-1] == 3:
        alpha   = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
        mask_np = np.concatenate([mask_np, alpha], axis=-1)
    bov_mask = np.all(mask_np == cor, axis=-1)
    out      = img_np.copy()
    out[~bov_mask] = 0
    return np.array(Image.fromarray(out).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))

if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    X = np.load(CACHE_X)
    y = np.load(CACHE_Y)
else:
    X_list, y_list = [], []
    for key, side_entry, rear_entry in pairs:
        try:
            side_px = segmentar(side_entry["img"], side_entry["mask"], side_entry["cor"])
            rear_px = segmentar(rear_entry["img"], rear_entry["mask"], rear_entry["cor"])
            combined = np.concatenate([
                side_px.flatten().astype(np.float32) / 255.0,
                rear_px.flatten().astype(np.float32) / 255.0,
            ])
            X_list.append(combined)
            y_list.append(side_entry["weight"])
        except Exception:
            continue
    X = np.array(X_list)
    y = np.array(y_list)
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)

print("\nSplit 80/20 + PCA ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

pca     = PCA(n_components=256, random_state=SEED)
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

y_pred = np.clip(mlp.predict(X_test), 50.0, WEIGHT_MAX)

mae       = mean_absolute_error(y_test, y_pred)
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
r2        = r2_score(y_test, y_pred)
pearson_r = np.corrcoef(y_test, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — MLP Multi-View (pixels {IMG_SIZE}x{IMG_SIZE} side+rear)
{'='*60}

    print("\nResultados:")
    print(f"MAE  (Erro Absoluto Médio)  : {mae:>8.2f} kg")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg")
    print(f"R²  (Coef. de Determinação)  : {r2:>8.4f}")
    print(f"Pearson r  : {pearson_r:>8.4f}")
""")

print("Gerando gráficos ...")
DPI    = 150
errors = y_pred - y_test

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(mlp.loss_curve_, color="#4A90D9", linewidth=2, label="Loss Treino")
ax.set_title("MLP Multi-View — Curva de Loss", fontsize=13, fontweight="bold")
ax.set_xlabel("Iteração"); ax.set_ylabel("Loss (MSE)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "mlp_multiview_curva_loss.png"), dpi=DPI)
plt.close()
print("    [OK] mlp_multiview_curva_loss.png")

fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#4A90D9", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"MLP Multi-View — Real vs Previsto  (R²={r2:.4f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "mlp_multiview_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    [OK] mlp_multiview_dispersao_r2.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#4A90D9", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("MLP Multi-View — Histograma de Erros", fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "mlp_multiview_histograma_erros.png"), dpi=DPI)
plt.close()
print("    [OK] mlp_multiview_histograma_erros.png")

from utils_tcc import safe_path, salvar_json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
np.save(safe_path(os.path.join(SAVE_DIR, "predictions_mlp_multiview.npy")), np.array([y_test, y_pred]))
salvar_json(RESULTS_PATH, "MLP Multi-View", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
    "train_history": mlp.loss_curve_, "val_history": [],
})
print("\n  Resultados salvos em resultados_modelos.json\n")
