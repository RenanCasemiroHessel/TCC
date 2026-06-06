#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Execução completa (Single-View + Multi-View + Comparação)
# Uso: bash rodar_tudo.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
PYTHON="./.venv/bin/python"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   TCC FEI — Execução Completa (todos os modelos)         ║"
echo "║   Dataset: sem B3, apenas leves (<250kg)                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "━━━ [0/9] Filtrando dataset ━━━"
$PYTHON filtrar_dataset.py
echo ""

echo "━━━ [1/9] Visualização das imagens segmentadas ━━━"
$PYTHON visualizar_segmentacao.py
echo ""


echo "━━━ [2/9] Regressão Linear  (~1 min) ━━━"
$PYTHON train_linear_regression.py
echo ""

echo "━━━ [3/9] XGBoost  (~1 min) ━━━"
$PYTHON train_xgboost.py
echo ""

echo "━━━ [4/9] MLP pixels  (~20 min) ━━━"
$PYTHON train_mlp.py
echo ""

echo "━━━ [5/9] CNN ResNet18 single-view  (~50 min) ━━━"
$PYTHON treinar_cnn.py
echo ""


echo "━━━ [6/9] LR Multi-View  (~1 min) ━━━"
$PYTHON train_lr_multiview.py
echo ""

echo "━━━ [7/9] XGBoost Multi-View  (~1 min) ━━━"
$PYTHON train_xgboost_multiview.py
echo ""

echo "━━━ [8/9] MLP Multi-View  (~20 min) ━━━"
$PYTHON train_mlp_multiview.py
echo ""
    
echo "━━━ [8/9] CNN Multi-View  (~50 min) ━━━"
$PYTHON treinar_cnn_multiview.py
echo ""


echo "━━━ [9/9] Comparando todos os modelos ━━━"
$PYTHON comparar_modelos.py
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Todos os modelos concluídos!                           ║"
echo "║   Gráficos salvos em: graficos/                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Gráficos gerados em graficos/:"
ls graficos/*.png 2>/dev/null | sed 's/^/    ✓ /' || echo "    (nenhum PNG encontrado)"
