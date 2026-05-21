#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TCC FEI — Treino da CNN (ResNet18) separado dos outros modelos
# Uso: bash rodar_cnn.sh
#      SAMPLE_FRAC=0.1 bash rodar_cnn.sh   (teste rápido com 10% dos dados)
# ─────────────────────────────────────────────────────────────────────────────

set -e
PYTHON="./.venv/bin/python"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   TCC FEI — CNN ResNet18 (35 épocas, GPU)               ║"
echo "║   Dataset: sem B3, apenas leves (<250kg)                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

$PYTHON treinar_cnn.py

echo ""
echo "━━━ Atualizando comparação de modelos ━━━"
$PYTHON comparar_modelos.py

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   CNN concluída! Modelo salvo: resnet18_gado_segmentado.pth ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
