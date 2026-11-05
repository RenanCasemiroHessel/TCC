"""
Teste de extrapolação: o ensemble single-view v2 em animais ACIMA da faixa de
treino (> 430kg, ficaram fora do cache e do k-fold).

Mostra quanto o modelo erra fora do que conhece — argumento pro aviso de
"fora da faixa" no app e pra delimitar no TCC até onde o modelo vale.

Uso (depois do treino):
  python testar_extrapolacao.py
  python testar_extrapolacao.py --modelos modelos/modelos_singleview_v2_sticker
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from inferencia_singleview import EnsembleSingleView
from pipeline_singleview import mascaras

from caminhos import DADOS, MODELOS

PASTA = os.path.join(DADOS, "teste_extrapolacao")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default=os.path.join(MODELOS, "modelos_final_singleview"))
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(PASTA, "pesados_acima_430kg.csv"))
    ens = EnsembleSingleView(args.modelos)

    print(f"  Ensemble de {len(ens.membros)} modelos em {os.path.basename(args.modelos)}\n")
    print(f"  {'imagem':32s} {'real':>6s} {'predito':>8s} {'erro':>8s} {'desvio modelos':>15s}")
    erros = []
    for r in df.itertuples():
        img = np.array(ImageOps.exif_transpose(Image.open(os.path.join(PASTA, r.imagem))).convert("RGB"))
        cattle, sticker = mascaras(np.array(Image.open(os.path.join(PASTA, r.mascara)).convert("RGB")), r.lote)
        try:
            p = ens.prever(img, cattle, 1.0 if r.sexo == "M" else 0.0, sticker)
        except ValueError as e:  # modelo com sticker em foto traseira
            print(f"  {os.path.basename(r.imagem):32s} {r.peso_real_kg:6.0f}   pulada: {e}")
            continue
        erro = p["peso_kg"] - r.peso_real_kg
        erros.append(erro)
        print(f"  {os.path.basename(r.imagem):32s} {r.peso_real_kg:6.0f} {p['peso_kg']:8.1f} {erro:+8.1f} "
              f"{p['desvio_entre_modelos_kg']:15.1f}")

    if erros:
        erros = np.array(erros)
        print(f"\n  MAE fora da faixa: {np.abs(erros).mean():.1f} kg | viés médio: {erros.mean():+.1f} kg "
              f"(negativo = subestima)")


if __name__ == "__main__":
    main()
