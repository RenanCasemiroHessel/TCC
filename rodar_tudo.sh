#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TCC FEI — Execução completa (do mais rápido para o mais lento)
# Uso: bash rodar_tudo.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
PYTHON="./.venv/bin/python"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   TCC FEI — Execução Completa de Todos os Modelos       ║"
echo "║   Ordem: Rápido → Lento                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 0. Visualização da Segmentação ───────────────────────────────────────────
echo "━━━ [0/6] Visualização das imagens segmentadas ━━━"
$PYTHON visualizar_segmentacao.py
echo "      → Abra 'visualizacao_segmentacao.png'!"
echo ""

# ── 1. Regressão Linear (~1 min) ─────────────────────────────────────────────
echo "━━━ [1/6] Regressão Linear  (~1 min) ━━━"
$PYTHON train_linear_regression.py
echo ""

# ── 2. XGBoost (~2 min) ──────────────────────────────────────────────────────
echo "━━━ [2/6] XGBoost  (~2 min) ━━━"
$PYTHON train_xgboost.py
echo ""

# ── 3. XGBoost + GridSearch (~15 min) ────────────────────────────────────────
echo "━━━ [3/6] XGBoost + GridSearch  (~15 min) ━━━"
$PYTHON tune_xgboost.py
echo ""

# ── 4. MLP com pixels (~20 min) ──────────────────────────────────────────────
echo "━━━ [4/6] MLP — Pixels achatados 64x64  (~20 min) ━━━"
$PYTHON train_mlp.py
echo ""

# ── 5. CNN ResNet18 (~2-3h com 15 épocas na GPU) ─────────────────────────────
echo "━━━ [5/6] CNN — ResNet18 + segmentação (GPU, ~2h) ━━━"
$PYTHON treinar_cnn.py
echo ""

# ── 6. Comparação de todos os modelos ────────────────────────────────────────
echo "━━━ [6/6] Comparando todos os modelos ━━━"
$PYTHON comparar_modelos.py
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Todos os modelos concluídos!                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Gráficos gerados:"
ls *.png 2>/dev/null | sed 's/^/    ✓ /'
