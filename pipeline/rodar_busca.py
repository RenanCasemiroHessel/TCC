"""
Busca automática de configuração da CNN single-view, em rodadas.

Cada configuração roda treinar_cnn_singleview_v2.py com 3 folds no conjunto de
DESENVOLVIMENTO (o teste final de 15% fica de fora o tempo todo). A escolha é
pelo MAE out-of-fold; o teste final só é medido uma vez, no modelo final.

  Rodada 1 — backbone (224px): resnet18, convnext_tiny, resnet50, efficientnet_v2_s
  Rodada 2 — resolução (320, 384px) no melhor backbone
  Rodada 3 — ajustes no melhor backbone (224px, mais rápido): repetição dos pesados,
             augmentação forte, EMA, mixup, LR do backbone
  Final    — melhor backbone + melhor resolução + ajustes que melhoraram o MAE,
             5 folds, medido no teste final; depois o empilhamento (CPU)

Retomável: configuração com resultados.json pronto é pulada; uma interrompida
continua do último checkpoint. Resumo em <busca>/tabela.csv e <busca>/resumo.txt.

Uso:
  python rodar_busca.py
  python rodar_busca.py --epocas 1 --folds-busca 2 --folds-final 2 --busca /tmp/x --final /tmp/x/final \\
      --cache-224 ... --cache-grande ...       # teste rápido
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

from caminhos import DADOS, MODELOS

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BACKBONES = ["resnet18", "convnext_tiny", "resnet50", "efficientnet_v2_s"]
BATCH = {224: 32, 320: 24, 384: 16}
AJUSTES = {
    "pesado3": {"pesado-repeticoes": 3},
    "pesado1": {"pesado-repeticoes": 1},
    "aug_forte": {"aug": "forte"},
    "ema": {"ema": 0.999},
    "mixup": {"mixup": 0.2},
    "lr_backbone_1e-4": {"lr-backbone": 1e-4},
}
MARGEM_GANHO = 0.15   # kg — ajuste só entra no final se melhorar o MAE pelo menos isso


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busca", default=os.path.join(MODELOS, "busca_singleview"))
    ap.add_argument("--final", default=os.path.join(MODELOS, "modelos_final_singleview"))
    ap.add_argument("--cache-224", default=os.path.join(DADOS, "cache_singleview_256", "dataset_singleview_cache.csv"))
    ap.add_argument("--cache-grande", default=os.path.join(DADOS, "cache_singleview_448", "dataset_singleview_cache.csv"))
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--folds-busca", type=int, default=3)
    ap.add_argument("--folds-final", type=int, default=5)
    return ap.parse_args()


def log(args, msg):
    print(msg, flush=True)
    with open(os.path.join(args.busca, "resumo.txt"), "a") as f:
        f.write(msg + "\n")


def flags(cfg):
    out = []
    for k, v in cfg.items():
        out += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    return out


def rodar(args, nome, cfg, saida, folds, extra=()):
    resultado = os.path.join(saida, "resultados.json")
    if os.path.exists(resultado):
        r = json.load(open(resultado))
        log(args, f"  [pulado, já existe] {nome:28s} MAE {r['oof']['mae']:6.2f} | ±10% {r['oof']['acc10']:5.1f}%")
        return r
    os.makedirs(saida, exist_ok=True)
    tam = int(cfg.get("img-size", 224))
    cmd = [PY, "-u", os.path.join(SAVE_DIR, "treinar_cnn_singleview_v2.py"),
           "--saida", saida, "--folds", str(folds), "--epocas", str(args.epocas), "--modo-sticker", "todos",
           "--cache-csv", args.cache_224 if tam <= 224 else args.cache_grande,
           "--batch", str(BATCH.get(tam, 16)), *flags(cfg), *extra]
    log(args, f"  >> {nome}: {' '.join(flags(cfg)) or '(padrão)'} — início {time.strftime('%H:%M')}")
    t0 = time.time()
    with open(os.path.join(saida, "log.txt"), "a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(resultado):
        log(args, f"  !! {nome} FALHOU (código {rc}) — ver {os.path.join(saida, 'log.txt')}")
        return None
    r = json.load(open(resultado))
    minutos = (time.time() - t0) / 60
    o = r["oof"]
    log(args, f"  ok {nome:28s} MAE {o['mae']:6.2f} | ±10% {o['acc10']:5.1f}% | R² {o['r2']:.3f} | {minutos:.0f} min")
    novo = not os.path.exists(os.path.join(args.busca, "tabela.csv"))
    with open(os.path.join(args.busca, "tabela.csv"), "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["config", "parametros", "mae", "acc10", "r2", "mape", "minutos"])
        w.writerow([nome, " ".join(flags(cfg)), f"{o['mae']:.3f}", f"{o['acc10']:.2f}", f"{o['r2']:.4f}",
                    f"{o['mape']:.2f}", f"{minutos:.1f}"])
    return r


def melhor(resultados):
    validos = {n: (cfg, r) for n, (cfg, r) in resultados.items() if r is not None}
    nome = min(validos, key=lambda n: validos[n][1]["oof"]["mae"])
    return nome, validos[nome][0], validos[nome][1]


def main():
    args = parse_args()
    os.makedirs(args.busca, exist_ok=True)
    log(args, f"\n=== Busca iniciada {time.strftime('%d/%m %H:%M')} ===")

    log(args, "\nRodada 1 — backbone (224px)")
    r1 = {}
    for bb in BACKBONES:
        cfg = {"backbone": bb}
        r1[f"r1_{bb}"] = (cfg, rodar(args, f"r1_{bb}", cfg, os.path.join(args.busca, f"r1_{bb}"), args.folds_busca))
    nome1, cfg1, res1 = melhor(r1)
    log(args, f"  -> melhor backbone: {cfg1['backbone']} (MAE {res1['oof']['mae']:.2f})")

    log(args, f"\nRodada 2 — resolução ({cfg1['backbone']})")
    r2 = {nome1: (cfg1, res1)}
    for tam in (320, 384):
        cfg = {**cfg1, "img-size": tam}
        nome = f"r2_{cfg1['backbone']}_{tam}"
        r2[nome] = (cfg, rodar(args, nome, cfg, os.path.join(args.busca, nome), args.folds_busca))
    nome2, cfg2, res2 = melhor(r2)
    log(args, f"  -> melhor resolução: {cfg2.get('img-size', 224)}px (MAE {res2['oof']['mae']:.2f})")

    log(args, f"\nRodada 3 — ajustes ({cfg1['backbone']}, 224px; referência MAE {res1['oof']['mae']:.2f})")
    ganhos = {}
    for nome_aj, aj in AJUSTES.items():
        cfg = {**cfg1, **aj}
        nome = f"r3_{nome_aj}"
        r = rodar(args, nome, cfg, os.path.join(args.busca, nome), args.folds_busca)
        if r is not None and r["oof"]["mae"] <= res1["oof"]["mae"] - MARGEM_GANHO:
            ganhos[nome_aj] = (aj, r["oof"]["mae"])
    # pesado3 e pesado1 mexem no mesmo parâmetro: fica só o melhor dos dois
    if "pesado3" in ganhos and "pesado1" in ganhos:
        ganhos.pop(max(("pesado3", "pesado1"), key=lambda k: ganhos[k][1]))
    log(args, f"  -> ajustes que melhoraram >= {MARGEM_GANHO} kg: {list(ganhos) or 'nenhum'}")

    cfg_final = dict(cfg2)
    for aj, _ in ganhos.values():
        cfg_final.update(aj)
    log(args, f"\nFinal — {args.folds_final} folds + teste final: {' '.join(flags(cfg_final))}")
    rf = rodar(args, "final", cfg_final, args.final, args.folds_final, extra=["--avaliar-holdout"])
    if rf is None:
        return
    h = rf.get("holdout", {})
    if h:
        log(args, f"  TESTE FINAL (ensemble): MAE {h['mae']:.2f} kg | ±10% {h['acc10']:.1f}% | R² {h['r2']:.3f}")

    empilhar = os.path.join(SAVE_DIR, "empilhamento.py")
    if os.path.exists(empilhar):
        log(args, "\nEmpilhamento (CPU)")
        with open(os.path.join(args.final, "empilhamento_log.txt"), "w") as f:
            subprocess.run([PY, "-u", empilhar, "--modelos", args.final], stdout=f, stderr=subprocess.STDOUT)
        log(args, f"  ver {os.path.join(args.final, 'empilhamento_log.txt')}")
    log(args, f"\n=== Busca concluída {time.strftime('%d/%m %H:%M')} ===")


if __name__ == "__main__":
    main()
