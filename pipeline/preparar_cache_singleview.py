"""
Pré-processa o dataset single-view UMA vez: segmenta com a máscara anotada,
recorta o boi mantendo a proporção, salva em 256px e calcula as features
escalares (tamanho relativo, sexo, sticker quando houver).

Antes cada época relia os JPGs em resolução cheia (1900x1425 + máscara PNG)
para cada amostra, o que dominava o tempo de treino. Depois disso o treino lê
só JPGs pequenos.

Fontes: dataset_pares_estendido.csv (B2+B4, lateral e traseira viram amostras
separadas) + dataset_singleview_b3.csv (B3, só lateral).

Uso:
  python preparar_cache_singleview.py                      # dataset completo
  python preparar_cache_singleview.py --limite 80 --saida /tmp/teste_cache
"""

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from caminhos import DADOS
from pipeline_singleview import (
    TAMANHO_CACHE, mascaras, escalares_mascara, recortar_boi, remover_sticker, sexo_do_arquivo,
)



def listar_imagens():
    pares = pd.read_csv(os.path.join(DADOS, "dataset_pares_estendido.csv"))
    b3 = pd.read_csv(os.path.join(DADOS, "dataset_singleview_b3.csv"))
    itens = []
    for _, r in pares.iterrows():
        itens.append((r["key"], r["side_path"], r["side_mask"], r["lote"], "side", r["weight"]))
        itens.append((r["key"], r["rear_path"], r["rear_mask"], r["lote"], "rear", r["weight"]))
    for _, r in b3.iterrows():
        itens.append((r["key"], r["path"], r["mask"], r["lote"], "side", r["weight"]))
    return itens


def processar(args):
    (key, img_path, mask_path, lote, vista, peso), pasta_imgs, tamanho = args
    sexo = sexo_do_arquivo(img_path)
    if sexo is None:
        return None
    try:
        # 83 JPGs (1,1%) têm tag EXIF de rotação; a máscara foi anotada na imagem já girada.
        # Sem exif_transpose os tamanhos não batem e os scripts antigos viravam imagem preta.
        img = np.array(ImageOps.exif_transpose(Image.open(img_path)).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
    except Exception:
        return None
    if img.shape[:2] != mask.shape[:2]:
        return None
    cattle, sticker = mascaras(mask, lote)
    if cattle.sum() < 1000:
        return None
    sticker_area = int(sticker.sum())
    img, corpo = remover_sticker(img, cattle, sticker)

    destino = os.path.join(pasta_imgs, f"{key}_{vista}.jpg")
    recortar_boi(img, corpo, tamanho).save(destino, quality=95)

    linha = {
        "key": key, "cache_path": destino, "peso_kg": float(peso),
        "lote": lote, "vista": vista, "sexo_M": sexo,
        "log_sticker_ratio": float(np.log(corpo.sum() / sticker_area)) if sticker_area >= 50 else np.nan,
    }
    linha.update(escalares_mascara(corpo))
    return linha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanho", type=int, default=TAMANHO_CACHE,
                    help="lado do recorte salvo em px (256 serve pra entrada 224; 448 pra 320/384)")
    ap.add_argument("--saida", default=None, help="padrão: cache_singleview_<tamanho>/")
    ap.add_argument("--limite", type=int, default=None, help="processa só N imagens sorteadas (teste)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--peso-max", type=float, default=430.0,
                    help="descarta animais acima disso (só 6 imagens, de 453 a 621kg, passam de 430)")
    args = ap.parse_args()
    if args.saida is None:
        args.saida = os.path.join(DADOS, f"cache_singleview_{args.tamanho}")

    pasta_imgs = os.path.join(args.saida, "imagens")
    os.makedirs(pasta_imgs, exist_ok=True)

    todos = listar_imagens()
    itens = [it for it in todos if it[5] <= args.peso_max]
    print(f"  Limite de peso: <= {args.peso_max:.0f}kg ({len(todos) - len(itens)} imagens acima descartadas)")
    if args.limite:
        itens = random.Random(0).sample(itens, args.limite)
    print(f"  Processando {len(itens)} imagens com {args.workers} processos -> {args.saida}")

    linhas, falhas = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(processar, [(it, pasta_imgs, args.tamanho) for it in itens], chunksize=8),
                                start=1):
            if res is None:
                falhas += 1
            else:
                linhas.append(res)
            if i % 500 == 0:
                print(f"    {i}/{len(itens)}")

    df = pd.DataFrame(linhas)
    csv_path = os.path.join(args.saida, "dataset_singleview_cache.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n  Válidas : {len(df)} | descartadas: {falhas}")
    print(f"  Vista   : {df['vista'].value_counts().to_dict()}")
    print(f"  Lote    : {df['lote'].value_counts().to_dict()}")
    print(f"  Sexo M  : {int(df['sexo_M'].sum())} | F: {int((1 - df['sexo_M']).sum())}")
    print(f"  Com sticker visível: {int(df['log_sticker_ratio'].notna().sum())}")
    print(f"  CSV     : {csv_path}")


if __name__ == "__main__":
    main()
