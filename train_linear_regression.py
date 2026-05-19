"""
TCC FEI — Regressão Linear para Estimativa de Peso Bovino
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

from sklearn.linear_model import LinearRegression
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
WEIGHT_MIN = 50.0
WEIGHT_MAX = 1000.0

np.random.seed(SEED)

print(f"\n{'='*60}")
print(f"  TCC FEI — Regressão Linear para Estimativa de Peso Bovino")
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

features = ["area", "altura_bbox", "largura_bbox", "proporcao", "area_norm"]
X = df[features].values
y = df["peso_kg"].values
print(f"      Features: {features}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT 80/20
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Dividindo dados (80% treino / 20% validação) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"      Treino: {len(X_train)} | Validação: {len(X_test)}")

# Normalização
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TREINO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Treinando Regressão Linear ...")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred       = model.predict(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# Classificação por faixas biológicas
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
  RELATÓRIO FINAL — Regressão Linear
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

  ┌─ COEFICIENTES DA REGRESSÃO ─────────────────────────────┐
""")
for feat, coef in zip(features, model.coef_):
    print(f"  │  {feat:<20}: {coef:>10.4f}                        │")
print(f"  │  Intercepto          : {model.intercept_:>10.4f}                        │")
print(f"  └──────────────────────────────────────────────────────────┘")

print("\n  Relatório de Classificação Detalhado:")
print(classification_report(y_true_cat, y_pred_cat,
                             labels=[0, 1, 2], target_names=class_names, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# 5. GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Gerando gráficos ...\n")
DPI      = 150
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
errors   = y_pred - y_test

# 5.1 Coeficientes da Regressão (equivalente à curva de loss)
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#E74C3C" if c < 0 else "#3498DB" for c in model.coef_]
ax.bar(features, model.coef_, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Regressão Linear — Coeficientes por Feature", fontsize=14, fontweight="bold")
ax.set_ylabel("Coeficiente (peso padronizado)")
ax.set_xlabel("Feature")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lr_curva_loss.png"), dpi=DPI)
plt.close()
print("    ✓ lr_curva_loss.png  (coeficientes da regressão)")

# 5.2 Dispersão R²
fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#E67E22", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"Regressão Linear — Dispersão Real vs Previsto  (R²={r2:.4f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lr_grafico_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    ✓ lr_grafico_dispersao_r2.png")

# 5.3 Histograma de Erros
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#E67E22", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("Regressão Linear — Histograma de Erros (Previsto − Real)", fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lr_histograma_erros.png"), dpi=DPI)
plt.close()
print("    ✓ lr_histograma_erros.png")

# 5.4 Boxplot/Violino de Erros por Faixa
df_err = pd.DataFrame({
    "Erro":  errors,
    "Faixa": [class_names[c] for c in y_true_cat]
})
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.violinplot(data=df_err, x="Faixa", y="Erro", palette="Set2", ax=axes[0])
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Reg. Linear — Violino de Erros por Faixa")
axes[0].set_ylabel("Erro (kg)")
sns.boxplot(data=df_err, x="Faixa", y="Erro", palette="Set2", ax=axes[1])
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Reg. Linear — Boxplot de Erros por Faixa")
axes[1].set_ylabel("Erro (kg)")
plt.suptitle("Regressão Linear — Distribuição dos Erros por Faixa Biológica",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lr_boxplot_violino_erros.png"), dpi=DPI)
plt.close()
print("    ✓ lr_boxplot_violino_erros.png")

print(f"\n{'='*60}")
print(f"  Gráficos gerados em: {SAVE_DIR}")
print(f"{'='*60}\n")

import json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
existing = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        existing = json.load(f)
existing["Regressão Linear"] = {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
    "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
}
with open(RESULTS_PATH, "w") as f:
    json.dump(existing, f, indent=2)
print("  Resultados salvos em: resultados_modelos.json\n")

