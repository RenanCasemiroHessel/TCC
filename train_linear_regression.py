"""
Regressão Linear (Ridge + Polinomial) para Estimativa de Peso Bovino
Entrada: Features extraídas das imagens segmentadas + engenharia de features
Dataset: apenas leves (<250 kg), sem B3, sem duplicatas
Modelo: Ridge com PolynomialFeatures(grau=2) para capturar relações não-lineares
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

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CONFIG
SEED       = 42
POLY_DEGREE = 2      # PolynomialFeatures — captura interações quadráticas
RIDGE_ALPHA = 10.0   # Regularização Ridge — evita overfitting com muitas features
np.random.seed(SEED)

print(f"  Regressão Linear (Ridge + Polinomial)")
print(f"  Entrada: Features das imagens segmentadas (CSV filtrado)")
print(f"  Modelo : Ridge(alpha={RIDGE_ALPHA}) + PolynomialFeatures(degree={POLY_DEGREE})")

# 1. CARREGAR CSV DE FEATURES (já filtrado: sem B3, só leves <250kg)
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "dataset_features_leves.csv")
print("Carregando dataset de features ...")

if not os.path.exists(CSV_PATH):
    print(f"  ERRO: Arquivo não encontrado: {CSV_PATH}")
    print("  Execute primeiro: python filtrar_dataset.py")
    exit(1)

df = pd.read_csv(CSV_PATH)

# One-Hot Encoding de sexo e vista
df_enc = pd.get_dummies(df, columns=["sexo", "vista"], drop_first=True)
for col in ["sexo_M", "vista_side"]:
    if col in df_enc.columns:
        df_enc[col] = df_enc[col].astype(float)

# ENGENHARIA DE FEATURES (mesma do XGBoost — relações biológicas e físicas)
df_enc["bbox_area"]          = df_enc["altura_bbox"] * df_enc["largura_bbox"]
df_enc["fill_ratio"]         = df_enc["area"] / (df_enc["bbox_area"] + 1e-8)
df_enc["volume_approx"]      = df_enc["area"] ** 1.5   # peso ~ volume ~ área^1.5
df_enc["volume_norm_approx"] = df_enc["area_norm"] ** 1.5

features_base = ["area", "altura_bbox", "largura_bbox", "proporcao", "area_norm",
                 "bbox_area", "fill_ratio", "volume_approx", "volume_norm_approx"]
for col in ["sexo_M", "vista_side"]:
    if col in df_enc.columns:
        features_base.append(col)

X_base = df_enc[features_base].values
y      = df_enc["peso_kg"].values

# 2. SPLIT 80/20
print("\nDividindo dados (80% treino / 20% validação) ...")
X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=SEED
)

# Normalização antes de expandir polinomialmente (evita escala explosiva)
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train_base)
X_test_sc    = scaler.transform(X_test_base)

# Expansão polinomial grau 2 (captura relações quadráticas e cruzadas)
poly         = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
X_train      = poly.fit_transform(X_train_sc)
X_test       = poly.transform(X_test_sc)

# 3. TREINO (Ridge)
print("\nTreinando Ridge Regression ...")
model = Ridge(alpha=RIDGE_ALPHA)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred       = model.predict(X_test)

# 4. MÉTRICAS
mae_train = mean_absolute_error(y_train, y_pred_train)
mae       = mean_absolute_error(y_test, y_pred)
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
r2        = r2_score(y_test, y_pred)

# Correlação de Pearson entre real e previsto
pearson_r = np.corrcoef(y_test, y_pred)[0, 1]

print(f"""
{'='*60}
  RELATÓRIO FINAL — Regressão Linear (Ridge + Polinomial)
{'='*60}

    print("\nResultados:")
    print(f"MAE  Treino  : {mae_train:>8.2f} kg")
    print(f"MAE  Validação  : {mae:>8.2f} kg")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:>8.2f} kg")
    print(f"R²  (Coef. de Determinação)  : {r2:>8.4f}")
    print(f"Pearson r  : {pearson_r:>8.4f}")
""")

# 5. GRÁFICOS
print("Gerando gráficos ...\n")
DPI    = 150
errors = y_pred - y_test

# 5.1 MAE Treino vs Validação (substituição da curva de loss — LR não tem iterações)
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["MAE Treino", "MAE Validação"], [mae_train, mae],
              color=["#3498DB", "#E74C3C"], edgecolor="white", width=0.4)
for bar, val in zip(bars, [mae_train, mae]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.2f} kg",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_title("Regressão Linear — MAE Treino vs Validação", fontsize=13, fontweight="bold")
ax.set_ylabel("MAE (kg)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "lr_curva_loss.png"), dpi=DPI)
plt.close()
print("    [OK] lr_curva_loss.png  (MAE treino vs validação)")

# 5.2 Dispersão R²
fig, ax = plt.subplots(figsize=(8, 8))
lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax.scatter(y_test, y_pred, alpha=0.5, color="#E67E22", edgecolors="white", linewidths=0.4, s=40)
ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão Perfeita")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"Regressão Linear — Dispersão Real vs Previsto  (R²={r2:.4f})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "lr_grafico_dispersao_r2.png"), dpi=DPI)
plt.close()
print("    [OK] lr_grafico_dispersao_r2.png")

# 5.3 Histograma de Erros
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color="#E67E22", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.axvline(errors.mean(), color="#2C3E50", linestyle="-", linewidth=1.5,
           label=f"Média: {errors.mean():.1f} kg")
ax.set_title("Regressão Linear — Histograma de Erros (Previsto − Real)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Erro (kg)"); ax.set_ylabel("Frequência")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "lr_histograma_erros.png"), dpi=DPI)
plt.close()
print("    [OK] lr_histograma_erros.png")

# 5.4 Gráfico de Correlação Linear (Real vs Previsto com linha de tendência)
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred, alpha=0.5, color="#E67E22", edgecolors="white",
           linewidths=0.3, s=40, label="Amostras")
coef_corr = np.polyfit(y_test, y_pred, 1)
x_line = np.linspace(y_test.min(), y_test.max(), 100)
ax.plot(x_line, np.polyval(coef_corr, x_line), color="#E74C3C", linewidth=2,
        label=f"Tendência (r = {pearson_r:.4f})")
ax.plot(x_line, x_line, "k--", linewidth=1.2, alpha=0.5, label="y = x (perfeito)")
ax.set_title(f"Regressão Linear — Correlação Linear\nPearson r = {pearson_r:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Peso Real (kg)"); ax.set_ylabel("Peso Previsto (kg)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "lr_correlacao_linear.png"), dpi=DPI)
plt.close()
print("    [OK] lr_correlacao_linear.png")

print(f"  Gráficos gerados em: {GRAFICOS_DIR}")

from utils_tcc import safe_path, salvar_json
RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")
npy_path = safe_path(os.path.join(SAVE_DIR, "predictions_lr.npy"))
np.save(npy_path, np.array([y_test, y_pred]))
salvar_json(RESULTS_PATH, "Regressão Linear", {
    "mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(pearson_r),
})
print("  Resultados salvos em: resultados_modelos.json\n")
