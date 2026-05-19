"""
TCC FEI — Comparação de Todos os Modelos
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
import seaborn as sns

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 150

RESULTS_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")

print("\n" + "="*60)
print("  TCC FEI — Comparação de Todos os Modelos")
print("="*60 + "\n")

if not os.path.exists(RESULTS_PATH):
    print(f"  ERRO: Arquivo '{RESULTS_PATH}' não encontrado.")
    print("  Execute todos os modelos antes de rodar este script.")
    print("  Os modelos salvam seus resultados automaticamente neste arquivo.\n")
    exit(1)

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

print(f"  Modelos encontrados: {list(results.keys())}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. TABELA DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
modelos  = []
maes     = []
rmses    = []
r2s      = []
accs     = []
f1s      = []

for nome, m in results.items():
    modelos.append(nome)
    maes.append(m.get("mae", 0))
    rmses.append(m.get("rmse", 0))
    r2s.append(m.get("r2", 0))
    accs.append(m.get("accuracy", 0) * 100)
    f1s.append(m.get("f1", 0) * 100)

df_res = pd.DataFrame({
    "Modelo": modelos,
    "MAE (kg)": maes,
    "RMSE (kg)": rmses,
    "R²": r2s,
    "Acurácia (%)": accs,
    "F1-Score (%)": f1s,
}).set_index("Modelo")

print("  Tabela de Resultados:")
print(df_res.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. GRÁFICO: Comparação de MAE, RMSE e R² (histograma lado a lado)
# ─────────────────────────────────────────────────────────────────────────────
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
out = os.path.join(SAVE_DIR, "comparacao_modelos_regressao.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    ✓ comparacao_modelos_regressao.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRÁFICO: Comparação de Acurácia e F1-Score
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#F8F9FA")

axes[0].bar(modelos, accs, color=COLORS, edgecolor="white", linewidth=1.2)
axes[0].set_title("Acurácia por Faixa Biológica", fontsize=13, fontweight="bold")
axes[0].set_ylabel("%")
axes[0].set_ylim(0, 110)
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(accs):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(modelos, f1s, color=COLORS, edgecolor="white", linewidth=1.2)
axes[1].set_title("F1-Score Ponderado por Faixa Biológica", fontsize=13, fontweight="bold")
axes[1].set_ylabel("%")
axes[1].set_ylim(0, 110)
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(f1s):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

plt.suptitle("Comparação de Todos os Modelos — Métricas de Classificação (3 Faixas)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(SAVE_DIR, "comparacao_modelos_classificacao.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    ✓ comparacao_modelos_classificacao.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. GRÁFICO: Overfitting — Train vs Val Loss por época (CNN e MLP)
# ─────────────────────────────────────────────────────────────────────────────
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
            print(f"      [Info] {nome} não possui histórico de validação compatível. Plotando apenas treino.")

        ax.set_title(f"{nome} — Curva de Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Época / Iteração")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Análise de Overfitting — Train Loss vs Val Loss",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "analise_overfitting.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("    ✓ analise_overfitting.png")
else:
    print("    [!] Nenhum modelo com histórico de épocas encontrado para análise de overfitting.")

print(f"\n{'='*60}")
print(f"  Gráficos de comparação gerados em: {SAVE_DIR}")
print(f"{'='*60}\n")
