"""
Tabela comparativa a partir das predições out-of-fold já salvas — sem retreinar.

Lê os predicoes_oof*.npy da CNN e dos tabulares (com e sem sticker, os que
existirem) e calcula as mesmas métricas de regressão + classificação por
faixa (F1, precisão, recall) pra todos. Como os modelos de cada grupo usam os
mesmos folds e os mesmos dados, a comparação é direta dentro do grupo.

Uso:
  python comparar_metricas.py
  python comparar_metricas.py --detalhes                 # bloco completo de cada modelo
  python comparar_metricas.py --cortes 120 180 --limiar 250
"""

import argparse
import os

import numpy as np

from pipeline_singleview import (
    CORTES_FAIXAS, LIMIAR_PESADO, imprimir_resumo, metricas_classificacao, resumo_oof,
)

from caminhos import MODELOS as PASTA_MODELOS

MODELOS = {
    "com sticker (5.016 laterais)": {
        "CNN": "modelos_singleview_v2_sticker/predicoes_oof.npy",
        "Regressão Linear": "modelos_tabulares_v2_sticker/predicoes_oof_lr.npy",
        "XGBoost": "modelos_tabulares_v2_sticker/predicoes_oof_xgboost.npy",
    },
    "sem sticker (7.465 imagens)": {
        "CNN": "modelos_singleview_v2/predicoes_oof.npy",
        "Regressão Linear": "modelos_tabulares_v2/predicoes_oof_lr.npy",
        "XGBoost": "modelos_tabulares_v2/predicoes_oof_xgboost.npy",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortes", type=float, nargs="+", default=list(CORTES_FAIXAS),
                    help="limites entre faixas em kg (padrão: 130 190 = leve/médio/pesado do TCC1)")
    ap.add_argument("--limiar", type=float, default=LIMIAR_PESADO, help="peso mínimo de 'pesado' na detecção binária")
    ap.add_argument("--detalhes", action="store_true", help="imprime o bloco completo de cada modelo")
    args = ap.parse_args()
    cortes = tuple(sorted(args.cortes))

    for grupo, modelos in MODELOS.items():
        linhas = []
        for nome, caminho in modelos.items():
            caminho = os.path.join(PASTA_MODELOS, caminho)
            if not os.path.exists(caminho):
                continue
            real, pred = np.load(caminho)
            r = resumo_oof(real, pred)
            c = metricas_classificacao(real, pred, cortes, args.limiar)
            linhas.append((nome, r, c))
            if args.detalhes:
                print(f"\n{'=' * 60}\n  {nome} — {grupo}\n{'=' * 60}")
                r["classificacao"] = c
                imprimir_resumo(r)
        if not linhas:
            continue

        print(f"\n  {grupo} — faixas {'/'.join(f'{x:.0f}' for x in cortes)} kg, pesado >= {args.limiar:.0f} kg")
        print(f"  {'modelo':18s} {'MAE':>6s} {'R²':>6s} {'F1 macro':>9s} {'acc bal':>8s} "
              f"{'prec pes':>9s} {'rec pes':>8s} {'F1 pes':>7s}")
        for nome, r, c in linhas:
            p = c["pesados"]
            print(f"  {nome:18s} {r['mae']:6.2f} {r['r2']:6.3f} {c['f1_macro']:9.3f} "
                  f"{c['acuracia_balanceada']:8.3f} {p['precisao']:9.3f} {p['recall']:8.3f} {p['f1']:7.3f}")


if __name__ == "__main__":
    main()
