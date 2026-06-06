"""
Comparação de Todos os Modelos
Lê os resultados salvos por cada modelo e gera:
  1. Histograma comparativo de MAE, RMSE e R² entre todos os modelos
  2. Gráfico de overfitting (train vs val loss para CNN e MLP)
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,"figure.titlesize":17})
import seaborn as sns

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)
DPI = 150

RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")

print("\n" + "="*60)
print("  Comparação de Todos os Modelos")
print("="*60 + "\n")

if not os.path.exists(RESULTS_PATH):
    print(f"  ERRO: Arquivo '{RESULTS_PATH}' não encontrado.")
    print("  Execute todos os modelos antes de rodar este script.")
    print("  Os modelos salvam seus resultados automaticamente neste arquivo.\n")
    exit(1)

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

print(f"  Modelos encontrados: {list(results.keys())}\n")

# ── Filtro e reordenação dos modelos ─────────────────────────────────────────
# Remove XGBoost não-tuned caso exista; renomeia XGBoost (Tuned) → XGBoost
results.pop("XGBoost", None) if "XGBoost (Tuned)" in results else None
if "XGBoost (Tuned)" in results:
    results["XGBoost"] = results.pop("XGBoost (Tuned)")

# Garante ordem: single-view primeiro, depois multi-view
ORDEM = [
    "Regressão Linear", "XGBoost", "MLP (pixels)", "CNN (ResNet18)",
    "LR Multi-View", "XGBoost Multi-View", "MLP Multi-View", "CNN Multi-View",
]
results = {k: results[k] for k in ORDEM if k in results}

print(f"  Modelos após filtro : {list(results.keys())}\n")

# 1. TABELA DE MÉTRICAS
modelos    = []
maes       = []
rmses      = []
r2s        = []
pearson_rs = []

for nome, m in results.items():
    modelos.append(nome)
    maes.append(m.get("mae", 0))
    rmses.append(m.get("rmse", 0))
    r2s.append(m.get("r2", 0))
    pearson_rs.append(m.get("pearson_r", float("nan")))

df_res = pd.DataFrame({
    "Modelo":     modelos,
    "MAE (kg)":   maes,
    "RMSE (kg)":  rmses,
    "R²":         r2s,
    "Pearson r":  pearson_rs,
}).set_index("Modelo")

print("  Tabela de Resultados:")
print(df_res.to_string())
print()

# 2. GRÁFICO: Comparação de MAE, RMSE e R² (histograma lado a lado)
COLORS = plt.cm.Set2(np.linspace(0, 1, len(modelos)))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#F8F9FA")

# MAE
axes[0].bar(modelos, maes, color=COLORS, edgecolor="white", linewidth=1.2)
axes[0].set_title("MAE — Erro Absoluto Médio", fontsize=13, fontweight="bold")
axes[0].set_ylabel("kg")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(maes):
    axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

# RMSE
axes[1].bar(modelos, rmses, color=COLORS, edgecolor="white", linewidth=1.2)
axes[1].set_title("RMSE — Raiz do Erro Quadrático", fontsize=13, fontweight="bold")
axes[1].set_ylabel("kg")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(rmses):
    axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

# R²
axes[2].bar(modelos, r2s, color=COLORS, edgecolor="white", linewidth=1.2)
axes[2].set_title("R² — Coeficiente de Determinação", fontsize=13, fontweight="bold")
axes[2].set_ylabel("R²")
axes[2].tick_params(axis="x", rotation=30)
axes[2].axhline(0, color="red", linestyle="--", linewidth=1)
for i, v in enumerate(r2s):
    offset = 0.01 if v >= 0 else -0.05
    axes[2].text(i, v + offset, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[2].grid(True, alpha=0.3, axis="y")

plt.suptitle("Comparação de Todos os Modelos — Métricas de Regressão",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(GRAFICOS_DIR, "comparacao_modelos_regressao.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    [OK] comparacao_modelos_regressao.png")

# 3. GRÁFICO: Boxplot Comparativo de Erros entre todos os modelos
PRED_FILES = {
    "Regressão Linear":    "predictions_lr.npy",
    "XGBoost":             "predictions_xgboost.npy",
    "MLP (pixels)":        "predictions_mlp.npy",
    "CNN (ResNet18)":      "predictions_cnn.npy",
    "LR Multi-View":       "predictions_lr_multiview.npy",
    "XGBoost Multi-View":  "predictions_xgboost_multiview.npy",
    "MLP Multi-View":      "predictions_mlp_multiview.npy",
    "CNN Multi-View":      "predictions_cnn_multiview.npy",
}

rows = []
for nome in modelos:
    fname = PRED_FILES.get(nome)
    if fname is None:
        continue
    fpath = os.path.join(SAVE_DIR, fname)
    if not os.path.exists(fpath):
        print(f"    [!] {fname} não encontrado — rode o modelo {nome} primeiro.")
        continue
    preds = np.load(fpath)
    y_real, y_prev = preds[0], preds[1]
    erros = y_prev - y_real
    for e in erros:
        rows.append({"Modelo": nome, "Erro (kg)": e})

if rows:
    df_box = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8F9FA")
    palette = {nome: color for nome, color in zip(modelos, COLORS)}
    sns.boxplot(data=df_box, x="Modelo", y="Erro (kg)", palette=palette,
                order=modelos, ax=ax, width=0.5,
                flierprops=dict(marker="o", markerfacecolor="grey",
                                markersize=3, alpha=0.5, linestyle="none"))
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Erro = 0")
    ax.set_title("Comparação de Todos os Modelos — Distribuição dos Erros",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Erro de Predição (kg)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(GRAFICOS_DIR, "comparacao_modelos_boxplot_erros.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("    [OK] comparacao_modelos_boxplot_erros.png")
else:
    print("    [!] Nenhum arquivo de predições encontrado para o boxplot comparativo.")

# 4. GRÁFICO: Overfitting — Train vs Val Loss por época (CNN e MLP)
modelos_com_historico = {
    nome: m for nome, m in results.items()
    if "train_history" in m and "val_history" in m
}

if modelos_com_historico:
    n = len(modelos_com_historico)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#F8F9FA")

    for ax, (nome, m) in zip(axes, modelos_com_historico.items()):
        train_h = m["train_history"]
        val_h   = m["val_history"]
        epochs  = range(1, len(train_h) + 1)

        ax.plot(epochs, train_h, color="#3498DB", linewidth=2, marker="o", markersize=4, label="Loss Treino")
        
        if val_h and len(val_h) == len(train_h):
            ax.plot(epochs, val_h,   color="#E74C3C", linewidth=2, marker="s", markersize=4,
                    linestyle="--", label="Loss Validação")

            # Sombra de diferença (overfitting gap)
            ax.fill_between(epochs, train_h, val_h,
                            where=[v > t for t, v in zip(train_h, val_h)],
                            alpha=0.15, color="#E74C3C", label="Overfitting")
            ax.fill_between(epochs, train_h, val_h,
                            where=[v <= t for t, v in zip(train_h, val_h)],
                            alpha=0.15, color="#2ECC71", label="Underfitting")

            # Detectar ponto de maior overfitting
            gaps = [abs(v - t) for t, v in zip(train_h, val_h)]
            pior_epoca = gaps.index(max(gaps)) + 1
            ax.axvline(pior_epoca, color="#E67E22", linestyle=":", linewidth=1.5,
                       label=f"Maior gap: época {pior_epoca}")
        else:

        ax.set_title(f"{nome} — Curva de Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Época / Iteração")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Análise de Overfitting — Train Loss vs Val Loss",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(GRAFICOS_DIR, "analise_overfitting.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("    [OK] analise_overfitting.png")
else:
    print("    [!] Nenhum modelo com histórico de épocas encontrado para análise de overfitting.")

print(f"  Gráficos de comparação gerados em: {SAVE_DIR}")
