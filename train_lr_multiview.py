"""
Regressão Linear Multi-View (Ridge + Polinomial)
Usa features de vista lateral + traseira do mesmo animal concatenadas.
"""

import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,"figure.titlesize":17})

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED        = 42
POLY_DEGREE = 2
RIDGE_ALPHA = 10.0
np.random.seed(SEED)

SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "dataset_features_leves.csv")

print(f"  Regressão Linear Multi-View")

print("Carregando e pareando features side+rear ...")
df = pd.read_csv(CSV_PATH)

df_side = df[df["vista"] == "side"].copy()
df_back = df[df["vista"] == "back"].copy()

FEAT = ["area", "altura_bbox", "largura_bbox", "proporcao", "area_norm"]
df_side = df_side.rename(columns={c: f"{c}_side" for c in FEAT})
df_back = df_back.rename(columns={c: f"{c}_back"  for c in FEAT})

df_paired = pd.merge(
    df_side[["id_animal", "peso_kg", "sexo"] + [f"{c}_side" for c in FEAT]],
    df_back[["id_animal"]                    + [f"{c}_back"  for c in FEAT]],
    on="id_animal"
).dropna().drop_duplicates(subset=["id_animal"]).reset_index(drop=True)

df_paired["sexo_M"] = (df_paired["sexo"] == "M").astype(float)

for v in ["side", "back"]:
    df_paired[f"bbox_area_{v}"]     = df_paired[f"altura_bbox_{v}"] * df_paired[f"largura_bbox_{v}"]
    df_paired[f"fill_ratio_{v}"]    = df_paired[f"area_{v}"] / (df_paired[f"bbox_area_{v}"] + 1e-8)
    df_paired[f"volume_approx_{v}"] = df_paired[f"area_{v}"] ** 1.5
    df_paired[f"vol_norm_{v}"]      = df_paired[f"area_norm_{v}"] ** 1.5

features = (
    [f"{c}_side" for c in FEAT] +
    [f"{c}_back"  for c in FEAT] +
    ["bbox_area_side", "fill_ratio_side", "volume_approx_side", "vol_norm_side",
     "bbox_area_back", "fill_ratio_back", "volume_approx_back", "vol_norm_back",
     "sexo_M"]
)

X = df_paired[features].values
y = df_paired["peso_kg"].values

print("Dividindo dados (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

poly    = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
X_train = poly.fit_transform(X_train_sc)
X_test  = poly.transform(X_test_sc)

print("Treinando Ridge Regression ...")
model = Ridge(alpha=RIDGE_ALPHA)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred       = model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae       = mean_absolute_error(y_test, y_pred)
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
r2        = r2_score(y_test, y_pred)
pearson_r = np.corrcoef(y_test, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — Regressão Linear Multi-View
{'='*60}

    print("\nResultados:")
    print(f"MAE  Treino  : {mae_train:>8.2f} kg")
    print(f"MAE  Validação  : {mae:>8.2f} kg")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg")
    print(f"R²  (Coef. de Determinação)  : {r2:>8.4f}")
    print(f"Pearson r  : {pearson_r:>8.4f}")
""")

print("Gerando gráficos ...")
DPI    = 150
errors = y_pred - y_test

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["MAE Treino", "MAE Validação"], [mae_train, mae],
              color=["#3498DB", "#E74C3C"], edgecolor="white", width=0.4)
for bar, val in zip(bars, [mae_train, mae]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.2f} kg",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_title("LR Multi-View — MAE Treino vs Validação", fontsize=13, fontweight="bold")
ax.set_ylabel("MAE (kg)"); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "lr_multiview_curva_loss.png"), dpi=DPI)
plt.close()
print("    [OK] lr_multiview_curva_loss.png")

fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#E67E22", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"LR Multi-View — Real vs Previsto  (R²={r2:.4f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "lr_multiview_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    [OK] lr_multiview_dispersao_r2.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#E67E22", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.axvline(errors.mean(), color="#2C3E50", linestyle="-", linewidth=1.5,
           label=f"Média: {errors.mean():.1f} kg")
ax.set_title("LR Multi-View — Histograma de Erros", fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "lr_multiview_histograma_erros.png"), dpi=DPI)
plt.close()
print("    [OK] lr_multiview_histograma_erros.png")

from utils_tcc import safe_path, salvar_json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
np.save(safe_path(os.path.join(SAVE_DIR, "predictions_lr_multiview.npy")), np.array([y_test, y_pred]))
salvar_json(RESULTS_PATH, "LR Multi-View", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
})
print("\n  Resultados salvos em resultados_modelos.json\n")
