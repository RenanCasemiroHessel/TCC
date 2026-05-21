"""
TCC FEI — XGBoost para Estimativa de Peso Bovino
Entrada: Features extraídas das imagens segmentadas (área, bbox, sexo, vista)
Dataset: apenas leves (<250 kg), sem B3, sem duplicatas
Hiperparâmetros: best params obtidos via GridSearch anterior
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
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Best params do GridSearch anterior
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
# Melhores hiperparâmetros encontrados pelo GridSearch:
BEST_PARAMS = {
    "n_estimators":     100,
    "max_depth":        3,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 1.0,
}

np.random.seed(SEED)

print(f"\n{'='*60}")
print(f"  TCC FEI — XGBoost para Estimativa de Peso Bovino")
print(f"{'='*60}")
print(f"  Entrada: Features das imagens segmentadas (CSV filtrado)")
print(f"  Params : {BEST_PARAMS}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAR CSV DE FEATURES (já filtrado: sem B3, só leves <250kg)
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "dataset_features_leves.csv")
print("[1/5] Carregando dataset de features ...")

if not os.path.exists(CSV_PATH):
    print(f"  ERRO: Arquivo não encontrado: {CSV_PATH}")
    print("  Execute primeiro: python filtrar_dataset.py")
    exit(1)

df = pd.read_csv(CSV_PATH)
print(f"      Registros carregados: {len(df)}")
print(f"      Lotes: {df['lote'].value_counts().to_dict()}")

# One-Hot Encoding de sexo e vista
df_enc = pd.get_dummies(df, columns=["sexo", "vista"], drop_first=True)
for col in ["sexo_M", "vista_side"]:
    if col in df_enc.columns:
        df_enc[col] = df_enc[col].astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# ENGENHARIA DE FEATURES (BIOLÓGICA E FÍSICA)
# ─────────────────────────────────────────────────────────────────────────────
df_enc["bbox_area"]          = df_enc["altura_bbox"] * df_enc["largura_bbox"]
df_enc["fill_ratio"]         = df_enc["area"] / (df_enc["bbox_area"] + 1e-8)
df_enc["volume_approx"]      = df_enc["area"] ** 1.5   # peso ~ volume ~ área^1.5
df_enc["volume_norm_approx"] = df_enc["area_norm"] ** 1.5

features = ["area", "altura_bbox", "largura_bbox", "proporcao", "area_norm",
            "bbox_area", "fill_ratio", "volume_approx", "volume_norm_approx"]
for col in ["sexo_M", "vista_side"]:
    if col in df_enc.columns:
        features.append(col)

X = df_enc[features].values
y = df_enc["peso_kg"].values
print(f"      Features originais + engenharia: {features}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT 80/20
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Dividindo dados (80% treino / 20% validação) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"      Treino: {len(X_train)} | Validação: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TREINO XGBOOST COM BEST PARAMS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Treinando XGBoost com best params ...")

model = xgb.XGBRegressor(
    **BEST_PARAMS,
    random_state=SEED,
    objective="reg:squarederror",
    eval_metric="rmse",
)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)
train_rmse = model.evals_result()["validation_0"]["rmse"]
val_rmse   = model.evals_result()["validation_1"]["rmse"]
print(f"      RMSE final treino: {train_rmse[-1]:.2f} kg")
print(f"      RMSE final val:    {val_rmse[-1]:.2f} kg")

y_pred = model.predict(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# Correlação de Pearson entre real e previsto
pearson_r = np.corrcoef(y_test, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — XGBoost
{'='*60}

  ┌─ MÉTRICAS DE REGRESSÃO ─────────────────────────────────┐
  │  MAE  (Erro Absoluto Médio)    : {mae:>8.2f} kg           │
  │  RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg           │
  │  R²   (Coef. de Determinação)  : {r2:>8.4f}               │
  │  Pearson r                     : {pearson_r:>8.4f}               │
  └──────────────────────────────────────────────────────────┘
""")

# ─────────────────────────────────────────────────────────────────────────────
# 5. GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Gerando gráficos ...\n")
DPI    = 150
errors = y_pred - y_test

# 5.1 Curva RMSE por nº de árvores
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_rmse, color="#4A90D9", linewidth=2, label="RMSE Treino")
ax.plot(val_rmse,   color="#E74C3C", linewidth=2, linestyle="--", label="RMSE Validação")
ax.set_title("XGBoost — Curva RMSE por Nº de Árvores", fontsize=14, fontweight="bold")
ax.set_xlabel("Nº de Árvores (Estimadores)")
ax.set_ylabel("RMSE (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_curva_loss.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_curva_loss.png")

# 5.2 Dispersão R²
fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#27AE60", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"XGBoost — Dispersão Real vs Previsto  (R²={r2:.4f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_grafico_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_grafico_dispersao_r2.png")

# 5.3 Histograma de Erros
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#27AE60", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("XGBoost — Histograma de Erros (Previsto − Real)", fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_histograma_erros.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_histograma_erros.png")

# 5.4 Gráfico de Correlação Linear
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred, alpha=0.5, color="#27AE60", edgecolors="white", linewidths=0.3, s=40, label="Amostras")
coef_corr = np.polyfit(y_test, y_pred, 1)
x_line = np.linspace(y_test.min(), y_test.max(), 100)
ax.plot(x_line, np.polyval(coef_corr, x_line), color="#E74C3C", linewidth=2,
        label=f"Tendência (r = {pearson_r:.4f})")
ax.plot(x_line, x_line, "k--", linewidth=1.2, alpha=0.5, label="y = x (perfeito)")
ax.set_title(f"XGBoost — Correlação Linear\nPearson r = {pearson_r:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_correlacao_linear.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_correlacao_linear.png")

# 5.6 Feature Importance
fig, ax = plt.subplots(figsize=(8, 5))
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)
ax.barh([features[i] for i in sorted_idx], importances[sorted_idx], color="#27AE60")
ax.set_title("XGBoost — Importância das Features", fontsize=13, fontweight="bold")
ax.set_xlabel("Importância")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_feature_importance.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_feature_importance.png")

print(f"\n{'='*60}")
print(f"  Gráficos gerados em: {GRAFICOS_DIR}")
print(f"{'='*60}\n")

from utils_tcc import safe_path, salvar_json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
npy_path = safe_path(os.path.join(SAVE_DIR, "predictions_xgboost.npy"))
np.save(npy_path, np.array([y_test, y_pred]))
salvar_json(RESULTS_PATH, "XGBoost", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
    "train_history": train_rmse, "val_history": val_rmse,
})
print("  Resultados salvos em: resultados_modelos.json\n")
