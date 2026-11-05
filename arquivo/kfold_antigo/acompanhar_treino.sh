#!/bin/bash
# Mostra o progresso do treino (fold/época atual) e estima o tempo restante
# com base no tempo médio das épocas já concluídas.
#
# Uso: ./acompanhar_treino.sh [caminho-do-log]

LOG="${1:-treino_multiview_kfold.log}"

if [ ! -f "$LOG" ]; then
    echo "Log não encontrado: $LOG"
    exit 1
fi

# lidos direto do cabeçalho do log, então funciona pra qualquer escala configurada
TOTAL_EPOCAS=$(grep -oP '(?<=Épocas/fold : )\d+' "$LOG" | head -1)
TOTAL_FOLDS=$(grep -oP '(?<=K-Folds     : )\d+' "$LOG" | head -1)
TOTAL_EPOCAS="${TOTAL_EPOCAS:-8}"
TOTAL_FOLDS="${TOTAL_FOLDS:-3}"
TOTAL_ETAPAS=$((TOTAL_EPOCAS * TOTAL_FOLDS))

echo "============================================================"
echo "  Progresso: $LOG  (config: ${TOTAL_FOLDS} folds x ${TOTAL_EPOCAS} épocas)"
echo "============================================================"

FOLD_ATUAL=$(grep -oP "(?<=── Fold )\d+(?=/${TOTAL_FOLDS})" "$LOG" | tail -1)
EPOCAS_NO_FOLD=$(awk -v f="$FOLD_ATUAL" -v te="$TOTAL_EPOCAS" '
  /── Fold/ { in_fold = ($0 ~ "Fold " f "/") }
  in_fold && $0 ~ ("Época [0-9]+/" te) { n++ }
  END { print n+0 }
' "$LOG")

FOLDS_COMPLETOS=$(( ${FOLD_ATUAL:-1} - 1 ))
ETAPAS_FEITAS=$(( FOLDS_COMPLETOS * TOTAL_EPOCAS + EPOCAS_NO_FOLD ))

echo "  Fold atual   : ${FOLD_ATUAL:-1}/$TOTAL_FOLDS"
echo "  Época atual  : ${EPOCAS_NO_FOLD:-0}/$TOTAL_EPOCAS (dentro do fold)"
echo "  Progresso    : $ETAPAS_FEITAS/$TOTAL_ETAPAS épocas totais ($(( ETAPAS_FEITAS * 100 / TOTAL_ETAPAS ))%)"
echo

# tempo médio por época, a partir das linhas "| Tempo: XXs" ja impressas
TEMPO_MEDIO=$(grep -oP '(?<=\| )\d+\.\d+(?=s(  \* melhor)?$)' "$LOG" | \
  awk '{sum+=$1; n++} END { if (n>0) printf "%.1f", sum/n; else print "" }')

if [ -n "$TEMPO_MEDIO" ] && [ "$ETAPAS_FEITAS" -gt 0 ]; then
    RESTANTES=$(( TOTAL_ETAPAS - ETAPAS_FEITAS ))
    SEGUNDOS_RESTANTES=$(awk -v t="$TEMPO_MEDIO" -v r="$RESTANTES" 'BEGIN { printf "%.0f", t * r }')
    HORAS=$(( SEGUNDOS_RESTANTES / 3600 ))
    MIN=$(( (SEGUNDOS_RESTANTES % 3600) / 60 ))
    echo "  Tempo médio/época : ${TEMPO_MEDIO}s"
    echo "  Estimativa restante: ${HORAS}h${MIN}min"
    echo "  Termina por volta de: $(date -d "+${SEGUNDOS_RESTANTES} seconds" '+%H:%M (%d/%m)')"
else
    echo "  (ainda sem épocas completas para estimar tempo restante)"
fi

echo
echo "  Últimas linhas do log:"
echo "------------------------------------------------------------"
tail -8 "$LOG"
