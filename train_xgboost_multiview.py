"""
TCC FEI — XGBoost Multi-View
Usa features de vista lateral + traseira do mesmo animal concatenadas.
"""

import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
BEST_PARAMS = {
    "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 1.0,
}
np.random.seed(SEED)

SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "dataset_features_leves.csv")

print(f"\n{'='*60}")
print(f"  TCC FEI — XGBoost Multi-View")
print(f"{'='*60}\n")

# ─── 1. Carregar e parear ─────────────────────────────────────────────────────
print("[1/4] Carregando e pareando features side+rear ...")
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

print(f"      Pares válidos: {len(df_paired)}")

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

# ─── 2. Split 80/20 ──────────────────────────────────────────────────────────
print("[2/4] Dividindo dados (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"      Treino: {len(X_train)} | Validação: {len(X_test)}")

# ─── 3. Treino XGBoost ───────────────────────────────────────────────────────
print("[3/4] Treinando XGBoost Multi-View ...")
model = xgb.XGBRegressor(
    **BEST_PARAMS, random_state=SEED,
    objective="reg:squarederror", eval_metric="rmse",
)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

train_rmse = model.evals_result()["validation_0"]["rmse"]
val_rmse   = model.evals_result()["validation_1"]["rmse"]
y_pred     = model.predict(X_test)

mae       = mean_absolute_error(y_test, y_pred)
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
r2        = r2_score(y_test, y_pred)
pearson_r = np.corrcoef(y_test, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — XGBoost Multi-View
{'='*60}

  ┌─ MÉTRICAS DE REGRESSÃO ─────────────────────────────────┐
  │  MAE  (Erro Absoluto Médio)    : {mae:>8.2f} kg           │
  │  RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg           │
  │  R²   (Coef. de Determinação)  : {r2:>8.4f}               │
  │  Pearson r                     : {pearson_r:>8.4f}               │
  └──────────────────────────────────────────────────────────┘
""")

# ─── 4. Gráficos ─────────────────────────────────────────────────────────────
print("[4/4] Gerando gráficos ...")
DPI    = 150
errors = y_pred - y_test

# Curva RMSE por árvore
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_rmse, color="#4A90D9", linewidth=2, label="RMSE Treino")
ax.plot(val_rmse,   color="#E74C3C", linewidth=2, linestyle="--", label="RMSE Validação")
ax.set_title("XGBoost Multi-View — RMSE por Nº de Árvores", fontsize=13, fontweight="bold")
ax.set_xlabel("Nº de Árvores"); ax.set_ylabel("RMSE (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_multiview_curva_loss.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_multiview_curva_loss.png")

# Dispersão R²
fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#27AE60", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"XGBoost Multi-View — Real vs Previsto  (R²={r2:.4f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_multiview_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_multiview_dispersao_r2.png")

# Histograma de erros
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#27AE60", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("XGBoost Multi-View — Histograma de Erros", fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_multiview_histograma_erros.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_multiview_histograma_erros.png")

# Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)
ax.barh([features[i] for i in sorted_idx], importances[sorted_idx], color="#27AE60")
ax.set_title("XGBoost Multi-View — Importância das Features", fontsize=13, fontweight="bold")
ax.set_xlabel("Importância"); ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_multiview_feature_importance.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_multiview_feature_importance.png")

# Salvar
from utils_tcc import safe_path, salvar_json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
np.save(safe_path(os.path.join(SAVE_DIR, "predictions_xgboost_multiview.npy")), np.array([y_test, y_pred]))
salvar_json(RESULTS_PATH, "XGBoost Multi-View", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
    "train_history": train_rmse, "val_history": val_rmse,
})
print("\n  Resultados salvos em resultados_modelos.json\n")
