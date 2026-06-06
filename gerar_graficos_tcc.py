"""
gerar_graficos_tcc.py
Regenera todos os graficos usados no texto do TCC com fontes maiores e legíveis.
Gera a partir dos arquivos .npy, .csv e .json já salvos — não precisa retreinar modelos.
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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ─── FONTES GLOBAIS ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":          15,
    "axes.titlesize":     17,
    "axes.labelsize":     15,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "legend.fontsize":    13,
    "figure.titlesize":   18,
    "axes.titleweight":   "bold",
})

SAVE_DIR    = os.path.dirname(os.path.abspath(__file__))
GRAFICOS_DIR = os.path.join(SAVE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)
DPI = 150

CSV_PATH  = os.path.join(SAVE_DIR, "dataset_features_leves.csv")
JSON_PATH = os.path.join(SAVE_DIR, "resultados_modelos.json")

print("\n" + "=" * 60)
print("  Gerando gráficos do TCC com fontes maiores")
print("=" * 60)

# 1. GRÁFICO DE AMOSTRAS POR LOTE  →  grafico-amostras-lote.png
print("\n[1] grafico-amostras-lote.png")
df = pd.read_csv(CSV_PATH)

contagem = (
    df.groupby(["lote", "vista"])
    .size()
    .reset_index(name="n")
)

totais = df.groupby("lote").size().reset_index(name="total")

lotes   = ["B2", "B4"]
side_ct = []
rear_ct = []
tot_ct  = []
for lote in lotes:
    s = contagem[(contagem["lote"] == lote) & (contagem["vista"] == "side")]["n"].sum()
    r = contagem[(contagem["lote"] == lote) & (contagem["vista"] == "rear")]["n"].sum()
    side_ct.append(s)
    rear_ct.append(r)
    tot_ct.append(s + r)

x   = np.arange(len(lotes))
w   = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

b1 = ax.bar(x - w/2, side_ct, w, label="Vista Lateral (Side)",  color="#4C72B0", edgecolor="white")
b2 = ax.bar(x + w/2, rear_ct, w, label="Vista Traseira (Rear)", color="#DD8452", edgecolor="white")

for bar, v in zip(b1, side_ct):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                str(int(v)), ha="center", va="bottom", fontsize=14, fontweight="bold")
for bar, v in zip(b2, rear_ct):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                str(int(v)), ha="center", va="bottom", fontsize=14, fontweight="bold")

for xi, tot in zip(x, tot_ct):
    ax.text(xi, max(side_ct[xi - x[0]] if xi == x[0] else 0,
                    rear_ct[xi - x[0]] if xi == x[0] else 0) + 80,
            f"Total: {int(tot)}", ha="center", va="bottom",
            fontsize=13, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(lotes)
ax.set_xlabel("Lote")
ax.set_ylabel("Número de Imagens")
ax.set_title("Distribuição de Amostras por Lote e Vista\n(após filtragem: 50 ≤ peso < 250 kg)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "grafico-amostras-lote.png"),
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    [OK] salvo")

# 2. DISTRIBUIÇÃO DE PESOS  →  grafico-distribuicao-pesos.png
print("[2] grafico-distribuicao-pesos.png")

cores_lote = {"B2": "#4C72B0", "B4": "#DD8452"}

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

for lote, cor in cores_lote.items():
    dados = df[df["lote"] == lote]["peso_kg"]
    media = dados.mean()
    ax.hist(dados, bins=25, alpha=0.65, color=cor,
            label=f"{lote} (n={len(dados)}, μ={media:.0f} kg)",
            edgecolor="white")

media_total = df["peso_kg"].mean()
ax.axvline(media_total, color="red", linestyle="--", linewidth=2.5,
           label=f"Média total: {media_total:.1f} kg")

ax.set_xlabel("Peso (kg)")
ax.set_ylabel("Frequência")
ax.set_title("Distribuição de Pesos — Dataset (50 ≤ peso < 250 kg)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "grafico-distribuicao-pesos.png"),
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    [OK] salvo")

# 3. BOXPLOT COMPARATIVO ERROS ABSOLUTOS  →  comparacao_boxplot_erros_absolutos.png
print("[3] comparacao_boxplot_erros_absolutos.png")

import json as _json
with open(JSON_PATH) as f:
    resultados = _json.load(f)

MODELOS_ORDEM = [
    ("Reg. Linear",      "predictions_lr.npy",                  "Single-View", "#4C72B0"),
    ("XGBoost",          "predictions_xgboost.npy",             "Single-View", "#4C72B0"),
    ("MLP",              "predictions_mlp.npy",                 "Single-View", "#4C72B0"),
    ("LR Multi-View",    "predictions_lr_multiview.npy",        "Multi-View",  "#DD8452"),
    ("XGB Multi-View",   "predictions_xgboost_multiview.npy",   "Multi-View",  "#DD8452"),
    ("MLP Multi-View",   "predictions_mlp_multiview.npy",       "Multi-View",  "#DD8452"),
    ("CNN Multi-View",   "predictions_cnn_multiview.npy",       "CNN",         "#55A868"),
]

grupos = []
erros_abs_list = []
cores_box = []
maes_dict = {}

for label, fname, grupo, cor in MODELOS_ORDEM:
    fpath = os.path.join(SAVE_DIR, fname)
    if not os.path.exists(fpath):
        print(f"    [!] {fname} não encontrado — pulando {label}")
        continue
    preds = np.load(fpath)
    y_real, y_pred = preds[0], preds[1]
    erros = np.abs(y_pred - y_real)
    grupos.append(label)
    erros_abs_list.append(erros)
    cores_box.append(cor)
    maes_dict[label] = erros.mean()

n = len(grupos)
fig, ax = plt.subplots(figsize=(max(14, n * 1.8), 7))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

bp = ax.boxplot(erros_abs_list, labels=grupos, patch_artist=True,
                widths=0.55, showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor="grey", linestyle="none"),
                medianprops=dict(color="white", linewidth=2.5))

for patch, cor in zip(bp["boxes"], cores_box):
    patch.set_facecolor(cor)
    patch.set_alpha(0.80)

# MAE como diamante
for i, (label, mae) in enumerate(maes_dict.items(), start=1):
    ax.plot(i, mae, "D", color="black", markersize=9, zorder=5,
            label="MAE" if i == 1 else "")
    ax.text(i, mae + 1.2, f"{mae:.1f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="black")

ax.axhline(20, color="crimson", linestyle="--", linewidth=1.8,
           label="Referência: 20 kg")

# Legenda de grupos
patch_sv  = mpatches.Patch(color="#4C72B0", alpha=0.8, label="Single-View")
patch_mv  = mpatches.Patch(color="#DD8452", alpha=0.8, label="Multi-View")
patch_cnn = mpatches.Patch(color="#55A868", alpha=0.8, label="CNN")
mae_mark  = plt.Line2D([0], [0], marker="D", color="w",
                        markerfacecolor="black", markersize=9, label="MAE")
ref_line  = plt.Line2D([0], [0], color="crimson", linestyle="--",
                        linewidth=1.8, label="Referência: 20 kg")
ax.legend(handles=[patch_sv, patch_mv, patch_cnn, mae_mark, ref_line],
          loc="upper right", fontsize=12)

ax.set_ylabel("Erro Absoluto (kg)")
ax.set_xlabel("")
ax.set_title("Comparação de Modelos — Distribuição do Erro Absoluto")
ax.tick_params(axis="x", rotation=15)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "comparacao_boxplot_erros_absolutos.png"),
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    [OK] salvo")

# 4. CNN MULTI-VIEW — DISPERSÃO R²  →  cnn_multiview_dispersao_r2.png
print("[4] cnn_multiview_dispersao_r2.png")

cnn_npy = np.load(os.path.join(SAVE_DIR, "predictions_cnn_multiview.npy"))
y_real_cnn, y_pred_cnn = cnn_npy[0], cnn_npy[1]

ss_res = np.sum((y_real_cnn - y_pred_cnn) ** 2)
ss_tot = np.sum((y_real_cnn - y_real_cnn.mean()) ** 2)
r2_cnn = 1 - ss_res / ss_tot

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

ax.scatter(y_real_cnn, y_pred_cnn, alpha=0.45, s=22,
           color="#5B7FBF", edgecolors="none")
lim = [50, 255]
ax.plot(lim, lim, "r--", linewidth=2, label="y = x (perfeito)")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Peso Real (kg)")
ax.set_ylabel("Peso Predito (kg)")
ax.set_title(f"CNN Multi-View — Real × Predito  |  R² = {r2_cnn:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "cnn_multiview_dispersao_r2.png"),
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    [OK] salvo")

# 5. CNN MULTI-VIEW — CURVA DE LOSS  →  cnn_multiview_curva_loss.png
print("[5] cnn_multiview_curva_loss.png")

cnn_res    = resultados.get("CNN Multi-View", {})
train_hist = cnn_res.get("train_history", [])
val_hist   = cnn_res.get("val_history", [])

if train_hist:
    epocas = range(1, len(train_hist) + 1)
    melhor = int(np.argmin(val_hist)) + 1 if val_hist else None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    ax.plot(epocas, train_hist, color="#3498DB", linewidth=2.5,
            marker="o", markersize=5, label="Loss Treino")
    if val_hist and len(val_hist) == len(train_hist):
        ax.plot(epocas, val_hist, color="#E74C3C", linewidth=2.5,
                marker="s", markersize=5, linestyle="--", label="Loss Validação")
        if melhor:
            ax.axvline(melhor, color="#27AE60", linestyle=":", linewidth=2,
                       label=f"Melhor época ({melhor})")

    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (Huber normalizado)")
    ax.set_title("CNN Multi-View — Curva de Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, "cnn_multiview_curva_loss.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("    [OK] salvo")
else:
    print("    [!] Histórico de loss não encontrado no JSON — pulando")

# 6. XGBOOST MV — FEATURE IMPORTANCE  →  xgboost_multiview_feature_importance.png
# (valores medidos no modelo treinado, conforme exibido no texto do TCC)
print("[6] xgboost_multiview_feature_importance.png")

# Valores extraídos do modelo treinado (ver Figura 11 do TCC)
features_imp = {
    "volume_approx":      0.021,
    "volume_norm_approx": 0.028,
    "altura_bbox":        0.041,
    "bbox_area":          0.050,
    "area":               0.060,
    "area_norm":          0.065,
    "fill_ratio":         0.072,
    "largura_bbox":       0.087,
    "proporcao":          0.100,
    "vista_side":         0.195,
    "sexo_M":             0.310,
}
feat_names = list(features_imp.keys())
feat_vals  = list(features_imp.values())

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")
colors_fi = ["#27AE60" if v < 0.15 else "#1A7A44" for v in feat_vals]
bars = ax.barh(feat_names, feat_vals, color=colors_fi, edgecolor="white")
for bar, val in zip(bars, feat_vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=13)
ax.set_xlabel("Importância (gain)")
ax.set_title("XGBoost Multi-View — Importância das Features")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, "xgboost_multiview_feature_importance.png"),
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    [OK] salvo")

# 7. PIPELINE TCC  →  pipeline-tcc.png
print("[7] pipeline-tcc.png")

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")

def caixa(ax, cx, cy, texto, cor_fundo, cor_texto="white", w=2.2, h=0.85):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=cor_fundo, edgecolor="white",
                          linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, texto, ha="center", va="center",
            fontsize=13, fontweight="bold", color=cor_texto,
            zorder=4, wrap=True,
            multialignment="center")

def seta(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.8, mutation_scale=18),
                zorder=2)

# Linha 1 — topo (y=3.8): entrada e segmentação
Y1 = 3.8
caixa(ax, 1.4,  Y1, "Imagem\nBruta",          "#2C3E50")
caixa(ax, 3.9,  Y1, "YOLO11n-seg\nSegmentação","#8E44AD")
caixa(ax, 6.4,  Y1, "Imagem\nSegmentada",      "#2980B9")
caixa(ax, 8.9,  Y1, "Extração de\nFeatures",   "#16A085")

seta(ax, 2.51, Y1, 2.79, Y1)
seta(ax, 5.01, Y1, 5.29, Y1)
seta(ax, 7.51, Y1, 7.79, Y1)

# Seta descendo de Features para modelos
seta(ax, 8.9, Y1 - 0.425, 8.9, 2.3)

# Linha 2 — base (y=1.5): modelos
Y2 = 1.5
xs_mod = [1.4, 3.6, 5.9, 8.2, 10.8]
labels_mod = [
    "Regressão\nLinear",
    "XGBoost",
    "MLP\n(pixels + PCA)",
    "CNN\nResNet18",
    "Peso\nEstimado (kg)",
]
cores_mod = ["#E67E22", "#E67E22", "#E67E22", "#C0392B", "#27AE60"]

for xi, label, cor in zip(xs_mod, labels_mod, cores_mod):
    caixa(ax, xi, Y2, label, cor)

# Seta de Features desce e vira para esquerda até primeiro modelo
seta(ax, 8.9, 2.3, 8.2, 1.925)

# Setas horizontais entre modelos (da direita para esquerda — todos recebem features)
for i in range(len(xs_mod) - 2, 0, -1):
    seta(ax, xs_mod[i] - 1.1, Y2, xs_mod[i] - 1.38, Y2)

# Seta do último modelo para Peso
seta(ax, 9.31, Y2, 9.59, Y2)

# Legenda leve
ax.text(6.5, 0.35, "Single-View e Multi-View: Regressão Linear, XGBoost, MLP, CNN ResNet18",
        ha="center", va="center", fontsize=11, color="#555555", style="italic")

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(GRAFICOS_DIR, "pipeline-tcc.png"),
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("    [OK] salvo")

# 8. ANÁLISE DE OVERFITTING  →  analise_overfitting.png
print("[8] analise_overfitting.png")

modelos_hist = {
    "CNN Multi-View": resultados.get("CNN Multi-View", {}),
    "MLP Multi-View": resultados.get("MLP Multi-View", {}),
}
modelos_hist = {k: v for k, v in modelos_hist.items()
                if v.get("train_history") and v.get("val_history")}

if modelos_hist:
    n = len(modelos_hist)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 6))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#F8F9FA")

    for ax, (nome, m) in zip(axes, modelos_hist.items()):
        train_h = m["train_history"]
        val_h   = m["val_history"]
        epocas  = range(1, len(train_h) + 1)

        ax.plot(epocas, train_h, color="#3498DB", linewidth=2.5,
                marker="o", markersize=5, label="Loss Treino")
        ax.plot(epocas, val_h, color="#E74C3C", linewidth=2.5,
                marker="s", markersize=5, linestyle="--", label="Loss Validação")
        ax.fill_between(epocas, train_h, val_h,
                        where=[v > t for t, v in zip(train_h, val_h)],
                        alpha=0.15, color="#E74C3C", label="Gap overfitting")
        melhor = int(np.argmin(val_h)) + 1
        ax.axvline(melhor, color="#27AE60", linestyle=":", linewidth=2,
                   label=f"Melhor checkpoint: época {melhor}")
        ax.set_title(f"{nome} — Curva de Loss")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

    plt.suptitle("Análise de Overfitting — Treino vs Validação")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, "analise_overfitting.png"),
                dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("    [OK] salvo")
else:
    print("    [!] Histórico não encontrado — pulando")

print(f"  Todos os gráficos gerados em: {GRAFICOS_DIR}")
