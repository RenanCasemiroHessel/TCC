"""
Empilhamento: Regressão Linear sobre [predição da CNN + features de silhueta/sticker].

A CNN e as medidas de silhueta erram de jeitos diferentes; um segundo modelo
simples combinando as duas costuma errar menos que a CNN sozinha (medido antes:
~0,7 kg de ganho no MAE).

Avaliação honesta:
  - desenvolvimento: o segundo modelo é treinado/avaliado nos MESMOS folds da CNN,
    usando só predições out-of-fold dela;
  - teste final: treinado em todo o desenvolvimento, aplicado na predição do
    ensemble da CNN nos animais do teste final.

Salva <modelos>/empilhador.pkl (usado depois na inferência do app).

Uso:
  python empilhamento.py --modelos modelos/modelos_final_singleview
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_singleview import ESCALAR_STICKER, TEM_STICKER, imprimir_resumo, resumo_oof  # noqa: E402

from caminhos import DADOS, MODELOS
FEATURES = ["log_cnn", "log_area_norm", "log_altura_rel", "log_largura_rel", "fill_ratio", "proporcao",
            "sexo_M", "vista_side", ESCALAR_STICKER, TEM_STICKER]


def novo_modelo():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(10))


def montar(pred_df, cache, preenchimento):
    df = pred_df.merge(cache, on=["key", "vista"], how="left", suffixes=("", "_cache"))
    df["log_cnn"] = np.log(df["peso_pred"])
    df["vista_side"] = (df["vista"] == "side").astype(float)
    df[TEM_STICKER] = df[ESCALAR_STICKER].notna().astype(float)
    df[ESCALAR_STICKER] = df[ESCALAR_STICKER].fillna(preenchimento)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default=os.path.join(MODELOS, "modelos_final_singleview"))
    ap.add_argument("--cache-csv", default=os.path.join(DADOS, "cache_singleview_256", "dataset_singleview_cache.csv"))
    args = ap.parse_args()

    cache = pd.read_csv(args.cache_csv).drop(columns=["peso_kg", "cache_path", "lote"])
    oof = pd.read_csv(os.path.join(args.modelos, "predicoes_oof.csv"))
    preenchimento = float(cache.loc[cache.key.isin(oof.key), ESCALAR_STICKER].median())
    dev = montar(oof, cache, preenchimento)

    # desenvolvimento: mesmos folds da CNN
    empilhado = np.empty(len(dev))
    for fold in sorted(dev["fold"].unique()):
        tr, te = dev["fold"] != fold, dev["fold"] == fold
        m = novo_modelo().fit(dev.loc[tr, FEATURES], np.log(dev.loc[tr, "peso_real"]))
        empilhado[te.to_numpy()] = np.exp(m.predict(dev.loc[te, FEATURES]))

    real = dev["peso_real"].to_numpy()
    r_cnn, r_emp = resumo_oof(real, dev["peso_pred"].to_numpy()), resumo_oof(real, empilhado)
    print(f"\n{'=' * 60}\n  DESENVOLVIMENTO (out-of-fold, {dev['fold'].nunique()} folds)\n{'=' * 60}")
    print(f"  CNN sozinha : MAE {r_cnn['mae']:.2f} | ±10% {r_cnn['acc10']:.1f}% | R² {r_cnn['r2']:.3f}")
    print(f"  Empilhado   : MAE {r_emp['mae']:.2f} | ±10% {r_emp['acc10']:.1f}% | R² {r_emp['r2']:.3f}\n")

    final = novo_modelo().fit(dev[FEATURES], np.log(dev["peso_real"]))
    with open(os.path.join(args.modelos, "empilhador.pkl"), "wb") as f:
        pickle.dump({"modelo": final, "features": FEATURES, "sticker_preenchimento": preenchimento}, f)

    saida = {"dev_cnn": r_cnn, "dev_empilhado": r_emp}
    caminho_hold = os.path.join(args.modelos, "predicoes_holdout.csv")
    if os.path.exists(caminho_hold):
        hold = montar(pd.read_csv(caminho_hold), cache, preenchimento)
        real_h = hold["peso_real"].to_numpy()
        pred_h = np.exp(final.predict(hold[FEATURES]))
        h_cnn, h_emp = resumo_oof(real_h, hold["peso_pred"].to_numpy()), resumo_oof(real_h, pred_h)
        print(f"{'=' * 60}\n  TESTE FINAL — {len(hold)} imagens nunca usadas\n{'=' * 60}")
        print(f"  CNN (ensemble) : MAE {h_cnn['mae']:.2f} | ±10% {h_cnn['acc10']:.1f}% | R² {h_cnn['r2']:.3f}")
        print(f"  Empilhado      : MAE {h_emp['mae']:.2f} | ±10% {h_emp['acc10']:.1f}% | R² {h_emp['r2']:.3f}\n")
        print("  Detalhe do empilhado no teste final:")
        imprimir_resumo(h_emp)
        saida.update(holdout_cnn=h_cnn, holdout_empilhado=h_emp)
        hold[["key", "vista", "lote", "peso_real"]].assign(peso_pred=pred_h).to_csv(
            os.path.join(args.modelos, "predicoes_holdout_empilhado.csv"), index=False)

    with open(os.path.join(args.modelos, "resultados_empilhamento.json"), "w") as f:
        json.dump(saida, f, indent=2)
    print(f"  Salvo: empilhador.pkl e resultados_empilhamento.json em {args.modelos}")


if __name__ == "__main__":
    main()
