"""
TCC FEI — XGBoost para Estimativa de Peso Bovino
Entrada: Features extraídas das imagens segmentadas (área, bbox, etc.)
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
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
WEIGHT_MIN = 50.0
WEIGHT_MAX = 1000.0

np.random.seed(SEED)

print(f"\n{'='*60}")
print(f"  TCC FEI — XGBoost para Estimativa de Peso Bovino")
print(f"{'='*60}")
print(f"  Entrada: Features das imagens segmentadas (CSV)")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAR CSV DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_features_completo.csv")
print("[1/5] Carregando dataset de features ...")

if not os.path.exists(CSV_PATH):
    print(f"  ERRO: Arquivo não encontrado: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)

# Filtro de domínio
before = len(df)
df = df[(df["peso_kg"] >= WEIGHT_MIN) & (df["peso_kg"] <= WEIGHT_MAX)].copy()
df = df.reset_index(drop=True)
print(f"      Registros após filtro [{WEIGHT_MIN}-{WEIGHT_MAX}kg]: {len(df)}  (removidos: {before - len(df)})")

# One-Hot Encoding
df_enc = pd.get_dummies(df, columns=["sexo", "vista"], drop_first=True)
for col in ["sexo_M", "vista_side"]:
    if col in df_enc.columns:
        df_enc[col] = df_enc[col].astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# ENGENHARIA DE FEATURES (BIOLÓGICA E FÍSICA)
# ─────────────────────────────────────────────────────────────────────────────
df_enc["bbox_area"] = df_enc["altura_bbox"] * df_enc["largura_bbox"]
df_enc["fill_ratio"] = df_enc["area"] / (df_enc["bbox_area"] + 1e-8)
df_enc["volume_approx"] = df_enc["area"] ** 1.5              # Escalonamento alométrico tridimensional (peso ~ volume ~ área ^ 1.5)
df_enc["volume_norm_approx"] = df_enc["area_norm"] ** 1.5

features = ["area", "altura_bbox", "largura_bbox", "proporcao", "area_norm", "bbox_area", "fill_ratio", "volume_approx", "volume_norm_approx"]
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
# 3. TREINO XGBOOST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Treinando XGBoost ...")
n_estimators = 300

model = xgb.XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    objective="reg:squarederror",
    eval_metric="rmse",
)

eval_results = {}
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

class_names = ["Leve (<250kg)", "Médio (250-450kg)", "Pesado (>450kg)"]
def classify(w):
    if w < 250:  return 0
    if w <= 450: return 1
    return 2

y_true_cat = [classify(v) for v in y_test]
y_pred_cat = [classify(v) for v in y_pred]

acc  = accuracy_score(y_true_cat, y_pred_cat)
prec = precision_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)
rec  = recall_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)
f1   = f1_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0)

print(f"""
{'='*60}
  RELATÓRIO FINAL — XGBoost
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
# 5. GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Gerando gráficos ...\n")
DPI      = 150
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
errors   = y_pred - y_test

# 5.1 Curva RMSE por n_estimadores (equivalente à curva de loss)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_rmse, color="#4A90D9", linewidth=2, label="RMSE Treino")
ax.plot(val_rmse,   color="#E74C3C", linewidth=2, linestyle="--", label="RMSE Validação")
ax.set_title("XGBoost — Curva RMSE por Nº de Árvores", fontsize=14, fontweight="bold")
ax.set_xlabel("Nº de Árvores (Estimadores)")
ax.set_ylabel("RMSE (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "xgboost_curva_loss.png"), dpi=DPI)
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
plt.savefig(os.path.join(SAVE_DIR, "xgboost_grafico_dispersao_r2.png"), dpi=DPI)
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
plt.savefig(os.path.join(SAVE_DIR, "xgboost_histograma_erros.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_histograma_erros.png")

# 5.4 Boxplot/Violino de Erros por Faixa
df_err = pd.DataFrame({
    "Erro":  errors,
    "Faixa": [class_names[c] for c in y_true_cat]
})
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.violinplot(data=df_err, x="Faixa", y="Erro", palette="Set2", ax=axes[0])
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("XGBoost — Violino de Erros por Faixa")
axes[0].set_ylabel("Erro (kg)")
sns.boxplot(data=df_err, x="Faixa", y="Erro", palette="Set2", ax=axes[1])
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("XGBoost — Boxplot de Erros por Faixa")
axes[1].set_ylabel("Erro (kg)")
plt.suptitle("XGBoost — Distribuição dos Erros por Faixa Biológica", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "xgboost_boxplot_violino_erros.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_boxplot_violino_erros.png")

# 5.5 Feature Importance
fig, ax = plt.subplots(figsize=(8, 5))
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)
ax.barh([features[i] for i in sorted_idx], importances[sorted_idx], color="#27AE60")
ax.set_title("XGBoost — Importância das Features", fontsize=13, fontweight="bold")
ax.set_xlabel("Importância")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "xgboost_feature_importance.png"), dpi=DPI)
plt.close()
print("    ✓ xgboost_feature_importance.png")

print(f"\n{'='*60}")
print(f"  Gráficos gerados em: {SAVE_DIR}")
print(f"{'='*60}\n")

import json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
existing = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        existing = json.load(f)
existing["XGBoost"] = {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
    "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
    "train_history": train_rmse, "val_history": val_rmse,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(existing, f, indent=2)
print("  Resultados salvos em: resultados_modelos.json\n")
