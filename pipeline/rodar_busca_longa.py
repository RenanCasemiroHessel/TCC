"""
Busca longa: continuação da rodar_busca.py a partir do modelo final atual
(efficientnet_v2_s, 384px, sem repetir os pesados).

Cada experimento muda UMA coisa em relação à base e roda com 3 folds no
conjunto de desenvolvimento (o teste final de 15% continua de fora).

  Rodada 4 — base + mudanças isoladas:
             backbones maiores a 384px, resolução 448px, cabeça (neurônios,
             dropout), função de perda, LRs, weight decay, paciência maior
  Rodada 5 — junta, de cada grupo, a mudança que melhorou o MAE (>= 0,15 kg)
  Final    — a melhor entre combinação e melhor mudança isolada, 5 folds +
             teste final, salvo em modelos_final_v2 (o modelo atual não é tocado)

Retomável como a rodar_busca.py (resultados em busca_longa/tabela.csv e resumo.txt).

Uso:
  python rodar_busca_longa.py
  python rodar_busca_longa.py --epocas 1 --folds-busca 2 --folds-final 2 --busca /tmp/x \\
      --final /tmp/x/final --cache-grande ...     # teste rápido
"""

import argparse
import os
import time

from caminhos import DADOS, MODELOS
from rodar_busca import MARGEM_GANHO, flags, log, rodar

BASE = {"backbone": "efficientnet_v2_s", "img-size": 384, "pesado-repeticoes": 1, "batch": 16}

# grupo -> {nome: mudanças}; do mesmo grupo só entra uma mudança na combinação
EXPERIMENTOS = {
    "arquitetura": {
        "convnext_tiny": {"backbone": "convnext_tiny"},
        "resnet50": {"backbone": "resnet50"},
        # batch menor: com 16 (e 12 no V2-M) estoura os 6 GB da GPU
        "efficientnet_v2_m": {"backbone": "efficientnet_v2_m", "batch": 8},
        "convnext_small": {"backbone": "convnext_small", "batch": 12},
        "img448": {"img-size": 448, "batch": 12},
    },
    "cabeca": {"cabeca512": {"cabeca": 512}, "cabeca128": {"cabeca": 128}},
    "dropout": {"dropout0.2": {"dropout": 0.2}, "dropout0.6": {"dropout": 0.6}},
    "perda": {"perda_l1": {"perda": "l1"}, "perda_mse": {"perda": "mse"}},
    "lr_cabeca": {"lr_cabeca_1e-3": {"lr-cabeca": 1e-3}},
    "lr_backbone": {"lr_backbone_1e-5": {"lr-backbone": 1e-5}, "lr_backbone_5e-5": {"lr-backbone": 5e-5}},
    "wd": {"wd_1e-2": {"wd": 1e-2}},
    "paciencia": {"paciencia12": {"paciencia": 12, "epocas": 40}},
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busca", default=os.path.join(MODELOS, "busca_longa"))
    ap.add_argument("--final", default=os.path.join(MODELOS, "modelos_final_v2"))
    ap.add_argument("--cache-224", default=os.path.join(DADOS, "cache_singleview_256", "dataset_singleview_cache.csv"))
    ap.add_argument("--cache-grande", default=os.path.join(DADOS, "cache_singleview_448", "dataset_singleview_cache.csv"))
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--folds-busca", type=int, default=3)
    ap.add_argument("--folds-final", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.busca, exist_ok=True)
    log(args, f"\n=== Busca longa iniciada {time.strftime('%d/%m %H:%M')} ===")

    log(args, f"\nRodada 4 — base ({' '.join(flags(BASE))}) + mudanças isoladas")
    base = rodar(args, "r4_base", BASE, os.path.join(args.busca, "r4_base"), args.folds_busca)
    if base is None:
        return
    ref = base["oof"]["mae"]

    escolhidos, melhor_isolado = {}, None   # grupo -> (nome, mudanças, mae)
    for grupo, exps in EXPERIMENTOS.items():
        for nome, mud in exps.items():
            cfg = {**BASE, **mud}
            r = rodar(args, f"r4_{nome}", cfg, os.path.join(args.busca, f"r4_{nome}"), args.folds_busca)
            if r is None:
                continue
            mae = r["oof"]["mae"]
            if melhor_isolado is None or mae < melhor_isolado[2]:
                melhor_isolado = (nome, cfg, mae)
            if mae <= ref - MARGEM_GANHO and (grupo not in escolhidos or mae < escolhidos[grupo][2]):
                escolhidos[grupo] = (nome, mud, mae)
    log(args, f"  -> referência {ref:.2f} | melhoraram >= {MARGEM_GANHO} kg: "
              f"{[f'{n} ({m:.2f})' for n, _, m in escolhidos.values()] or 'nenhuma'}")

    if not escolhidos:
        log(args, "  Nenhuma mudança superou a base: o modelo final atual continua sendo o melhor.")
        log(args, f"\n=== Busca longa concluída {time.strftime('%d/%m %H:%M')} ===")
        return

    cfg_final = dict(BASE)
    for _, mud, _ in escolhidos.values():
        cfg_final.update(mud)
    if len(escolhidos) > 1:
        cfg_final["batch"] = min(m.get("batch", BASE["batch"]) for _, m, _ in escolhidos.values())
        log(args, "\nRodada 5 — combinação das mudanças que melhoraram")
        combo = rodar(args, "r5_combinacao", cfg_final, os.path.join(args.busca, "r5_combinacao"), args.folds_busca)
        if combo is None or combo["oof"]["mae"] > melhor_isolado[2]:
            log(args, f"  -> combinação não superou a melhor mudança isolada ({melhor_isolado[0]}); fica ela")
            cfg_final = melhor_isolado[1]

    log(args, f"\nFinal — {args.folds_final} folds + teste final: {' '.join(flags(cfg_final))}")
    rf = rodar(args, "final_v2", cfg_final, args.final, args.folds_final, extra=["--avaliar-holdout"])
    if rf is not None and rf.get("holdout"):
        h = rf["holdout"]
        log(args, f"  TESTE FINAL (ensemble): MAE {h['mae']:.2f} kg | ±10% {h['acc10']:.1f}% | R² {h['r2']:.3f}")
    log(args, f"\n=== Busca longa concluída {time.strftime('%d/%m %H:%M')} ===")


if __name__ == "__main__":
    main()
