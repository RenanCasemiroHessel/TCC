"""
CNN Single-View v2 — K-Fold configurável, a partir do cache de preparar_cache_singleview.py.

Base (desde a primeira versão):
  - lê o cache já segmentado/recortado (sem achatar), com escalares de tamanho + sexo;
  - otimizador único com 2 grupos (backbone / cabeça) e cosine em forma fechada
    (o bug antigo travava o LR ao descongelar);
  - saída linear em log(peso) padronizado;
  - melhor época escolhida numa validação INTERNA (10% dos animais do treino do
    fold), com early stopping; o fold de teste só mede. Avaliação com TTA (flip).

Configurável por linha de comando (pra busca de hiperparâmetros):
  --backbone resnet18|resnet50|convnext_tiny|efficientnet_v2_s   --img-size 224|320|384
  --cabeca/--dropout  --lr-backbone/--lr-cabeca/--wd  --perda smoothl1|l1|mse
  --aug leve|forte  --mixup ALFA  --ema DECAY  --pesado-limiar/--pesado-repeticoes
  --modo-sticker todos|so|nao:
      todos = dataset completo; sticker entra quando aparece (senão ratio preenchido + tem_sticker=0)
      so    = só fotos com sticker visível
      nao   = ignora o sticker

Teste final (--holdout, padrão 15%): sorteado por animal com seed fixa sobre o
cache inteiro — os mesmos animais em qualquer configuração. Nunca entra no
k-fold. Com --avaliar-holdout, o ensemble dos folds é medido nele no fim; é o
número a apresentar (não foi usado pra escolher nada).

Checkpoints por fold em <saida>/: sv2_fold{N}.pth (melhor) e sv2_fold{N}_latest.pth
(retomada). Rodar de novo retoma; fold que parou por early stopping fica concluído.

Uso:
  python treinar_cnn_singleview_v2.py
  python treinar_cnn_singleview_v2.py --backbone convnext_tiny --img-size 384 --batch 16 --avaliar-holdout
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caminhos import DADOS, MODELOS
from pipeline_singleview import (
    BACKBONES, ESCALARES_BASE, ESCALAR_STICKER, MARGEM, TEM_STICKER,
    SingleViewNet, folds_por_animal, imprimir_resumo, metricas, prever_tta, resumo_oof,
    transform_treino, transform_avaliacao,
)


SEED = 42
SEED_HOLDOUT = 123
FINETUNE_EPOCH = 3        # backbone congelado nas épocas 1..2
LR_MIN_FRAC = 0.01        # LR final = 1% do inicial
FRAC_VAL_INTERNA = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-csv", default=None,
                    help="padrão: cache_singleview_256 (img-size <= 224) ou cache_singleview_448")
    ap.add_argument("--saida", default=None, help="padrão: modelos_singleview_v2_<modo-sticker>/")
    ap.add_argument("--modo-sticker", choices=["todos", "so", "nao"], default="todos")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="resnet18")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cabeca", type=int, default=256, help="neurônios da camada oculta da cabeça")
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr-cabeca", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--perda", choices=["smoothl1", "l1", "mse"], default="smoothl1")
    ap.add_argument("--aug", choices=["leve", "forte"], default="leve")
    ap.add_argument("--mixup", type=float, default=0.0, help="alfa da Beta; 0 = desligado")
    ap.add_argument("--ema", type=float, default=0.0, help="decay da média dos pesos; 0 = desligado")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--paciencia", type=int, default=8)
    ap.add_argument("--pesado-limiar", type=float, default=230.0)
    ap.add_argument("--pesado-repeticoes", type=int, default=10,
                    help="quantas vezes cada imagem pesada aparece por época (1 = sem repetir)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.15, help="fração de animais do teste final")
    ap.add_argument("--avaliar-holdout", action="store_true",
                    help="no fim, mede o ensemble dos folds no teste final")
    args = ap.parse_args()
    if args.cache_csv is None:
        tam = 256 if args.img_size <= 224 else 448
        args.cache_csv = os.path.join(DADOS, f"cache_singleview_{tam}", "dataset_singleview_cache.csv")
    if args.saida is None:
        args.saida = os.path.join(MODELOS, f"modelos_singleview_v2_{args.modo_sticker}")
    return args


class CacheDataset(Dataset):
    def __init__(self, df, escalares_cols, esc_mean, esc_std, mu, sd, transform):
        self.paths = df["cache_path"].tolist()
        self.pesos = df["peso_kg"].to_numpy(dtype=np.float32)
        self.esc = ((df[escalares_cols].to_numpy(dtype=np.float32) - esc_mean) / esc_std).astype(np.float32)
        self.alvo = ((np.log(self.pesos) - mu) / sd).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return img, torch.from_numpy(self.esc[i]), torch.tensor(self.alvo[i]), torch.tensor(self.pesos[i])


def preencher_sticker(df, preenchimento):
    """Modo 'todos': marca quem tem sticker e preenche o ratio de quem não tem."""
    df = df.copy()
    df[TEM_STICKER] = df[ESCALAR_STICKER].notna().astype(float)
    df[ESCALAR_STICKER] = df[ESCALAR_STICKER].fillna(preenchimento)
    return df


def lr_cosine(base, epoca, total):
    """Cosine em forma fechada: não depende de estado de scheduler (retomada segura)."""
    progresso = (epoca - 1) / max(1, total - 1)
    return base * (LR_MIN_FRAC + (1 - LR_MIN_FRAC) * 0.5 * (1 + math.cos(math.pi * progresso)))


@torch.no_grad()
def prever_kg(model, loader, mu, sd):
    model.eval()
    preds, reais = [], []
    for imgs, esc, _alvo, peso in loader:
        imgs, esc = imgs.to(DEVICE, non_blocking=True), esc.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            p = prever_tta(model, imgs, esc)
        preds.append(torch.exp(p.float().squeeze(1) * sd + mu).cpu().numpy())
        reais.append(peso.numpy())
    return np.concatenate(preds), np.concatenate(reais)


def criar_loader(ds, shuffle, batch):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4,
                      pin_memory=DEVICE.type == "cuda", persistent_workers=True)


def criar_criterio(nome):
    return {"smoothl1": nn.SmoothL1Loss(beta=1.0), "l1": nn.L1Loss(), "mse": nn.MSELoss()}[nome]


def treinar_fold(fold, treino_df, teste_df, escalares_cols, args, tamanho_cache):
    best_path = os.path.join(args.saida, f"sv2_fold{fold}.pth")
    latest_path = os.path.join(args.saida, f"sv2_fold{fold}_latest.pth")

    chaves = treino_df["key"].unique()
    ch_tr, ch_val = train_test_split(chaves, test_size=FRAC_VAL_INTERNA, random_state=SEED + fold)
    tr_df = treino_df[treino_df["key"].isin(ch_tr)].reset_index(drop=True)
    val_df = treino_df[treino_df["key"].isin(ch_val)].reset_index(drop=True)

    preenchimento = None
    if TEM_STICKER in escalares_cols:
        preenchimento = float(tr_df[ESCALAR_STICKER].median())
        tr_df, val_df, teste_df = (preencher_sticker(d, preenchimento) for d in (tr_df, val_df, teste_df))

    esc_mean = tr_df[escalares_cols].to_numpy(dtype=np.float32).mean(axis=0)
    esc_std = tr_df[escalares_cols].to_numpy(dtype=np.float32).std(axis=0) + 1e-6
    mu = float(np.log(tr_df["peso_kg"]).mean())
    sd = float(np.log(tr_df["peso_kg"]).std())

    def ds(df, tf):
        return CacheDataset(df, escalares_cols, esc_mean, esc_std, mu, sd, tf)

    tf_treino = transform_treino(args.img_size, forte=args.aug == "forte")
    partes = [ds(tr_df, tf_treino)]
    pesados = tr_df[tr_df["peso_kg"] >= args.pesado_limiar]
    if len(pesados) and args.pesado_repeticoes > 1:
        partes += [ds(pesados, tf_treino)] * (args.pesado_repeticoes - 1)
    treino_ld = criar_loader(ConcatDataset(partes), True, args.batch)
    val_ld = criar_loader(ds(val_df, transform_avaliacao(args.img_size)), False, args.batch)
    teste_ld = criar_loader(ds(teste_df, transform_avaliacao(args.img_size)), False, args.batch)

    model = SingleViewNet(len(escalares_cols), args.backbone, args.cabeca, args.dropout).to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone, "base_lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_cabeca, "base_lr": args.lr_cabeca},
    ], weight_decay=args.wd)
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
           if args.ema > 0 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    criterio = criar_criterio(args.perda)

    config_ck = {
        "escalares": escalares_cols, "esc_mean": esc_mean, "esc_std": esc_std, "mu": mu, "sd": sd,
        "sticker_preenchimento": preenchimento, "backbone": args.backbone, "cabeca": args.cabeca,
        "dropout": args.dropout, "img_size": args.img_size, "tamanho_cache": tamanho_cache, "margem": MARGEM,
    }
    estado = {"epoca": 0, "historico": {"train_loss": [], "val_mae": []},
              "melhor_val_mae": float("inf"), "melhor_epoca": 0, "sem_melhora": 0, "parou_cedo": False}

    if os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        if ema is not None and ck.get("ema_state_dict"):
            ema.load_state_dict(ck["ema_state_dict"])
        estado = ck["estado"]
        print(f"\n  ── Fold {fold} ── retomando após a época {estado['epoca']}")
    else:
        print(f"\n  ── Fold {fold} ── do zero | treino={len(tr_df)} val_interna={len(val_df)} "
              f"teste={len(teste_df)} imgs")

    concluido = estado["parou_cedo"] or estado["epoca"] >= args.epocas
    for epoca in range(estado["epoca"] + 1, args.epocas + 1):
        if concluido:
            break
        t0 = time.time()
        congelado = epoca < FINETUNE_EPOCH
        for p in model.backbone.parameters():
            p.requires_grad = not congelado
        for g in opt.param_groups:
            g["lr"] = lr_cosine(g["base_lr"], epoca, args.epocas)

        model.train()
        if congelado:
            model.backbone.eval()  # não deixa o BatchNorm se mexer com o backbone congelado
        soma, n = 0.0, 0
        for imgs, esc, alvo, _peso in treino_ld:
            imgs, esc = imgs.to(DEVICE, non_blocking=True), esc.to(DEVICE, non_blocking=True)
            alvo = alvo.to(DEVICE, non_blocking=True).unsqueeze(1)
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                idx = torch.randperm(imgs.size(0), device=DEVICE)
                imgs = lam * imgs + (1 - lam) * imgs[idx]
                esc = lam * esc + (1 - lam) * esc[idx]
                alvo = lam * alvo + (1 - lam) * alvo[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = criterio(model(imgs, esc), alvo)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update_parameters(model)
            soma += loss.item() * imgs.size(0)
            n += imgs.size(0)

        avaliado = ema.module if ema is not None else model
        pred_val, real_val = prever_kg(avaliado, val_ld, mu, sd)
        val_mae = float(np.abs(pred_val - real_val).mean())
        estado["historico"]["train_loss"].append(soma / n)
        estado["historico"]["val_mae"].append(val_mae)
        estado["epoca"] = epoca

        marca = ""
        if val_mae < estado["melhor_val_mae"]:
            estado.update(melhor_val_mae=val_mae, melhor_epoca=epoca, sem_melhora=0)
            torch.save({"model_state_dict": avaliado.state_dict(), **config_ck}, best_path)
            marca = "  * melhor"
        else:
            estado["sem_melhora"] += 1
            if estado["sem_melhora"] >= args.paciencia:
                estado["parou_cedo"] = True

        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
                    "estado": estado}, latest_path)

        print(f"    Época {epoca:02d}/{args.epocas} | loss {soma / n:.4f} | val MAE {val_mae:6.2f}kg | "
              f"lr {opt.param_groups[1]['lr']:.1e} | {time.time() - t0:.0f}s{marca}")
        if estado["parou_cedo"]:
            print(f"    early stopping: {args.paciencia} épocas sem melhorar")
            break

    if concluido:
        print(f"    já concluído ({estado['epoca']} épocas, melhor = {estado['melhor_epoca']}) — só avaliando")

    ck = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    pred, real = prever_kg(model, teste_ld, mu, sd)
    return {"historico": estado["historico"], "melhor_epoca": estado["melhor_epoca"], "pred": pred, "real": real}


def prever_holdout(args, hold_df):
    """Ensemble dos folds (média do log-peso) no teste final, com as estatísticas de cada fold."""
    logs = []
    for fold in range(1, args.folds + 1):
        ck = torch.load(os.path.join(args.saida, f"sv2_fold{fold}.pth"), map_location=DEVICE, weights_only=False)
        model = SingleViewNet(len(ck["escalares"]), ck["backbone"], ck["cabeca"], ck["dropout"],
                              pretrained=False).to(DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        df = hold_df
        if ck["sticker_preenchimento"] is not None:
            df = preencher_sticker(df, ck["sticker_preenchimento"])
        ds = CacheDataset(df, ck["escalares"], ck["esc_mean"], ck["esc_std"], ck["mu"], ck["sd"],
                          transform_avaliacao(ck["img_size"]))
        pred, real = prever_kg(model, criar_loader(ds, False, args.batch), ck["mu"], ck["sd"])
        logs.append(np.log(pred))
    return np.exp(np.mean(logs, axis=0)), real


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(args.saida, exist_ok=True)

    if not os.path.exists(args.cache_csv):
        sys.exit(f"  ERRO: {args.cache_csv} não existe. Rode antes: python preparar_cache_singleview.py")

    df_todo = pd.read_csv(args.cache_csv)
    tamanho_cache = Image.open(df_todo["cache_path"].iloc[0]).size[0]

    # teste final sorteado sobre TODOS os animais do cache -> os mesmos em qualquer configuração
    chaves = np.array(sorted(df_todo["key"].unique()))
    rng = np.random.default_rng(SEED_HOLDOUT)
    chaves_hold = set(rng.choice(chaves, size=int(round(args.holdout * len(chaves))), replace=False))

    escalares_cols = list(ESCALARES_BASE)
    df = df_todo
    if args.modo_sticker == "so":
        df = df[df[ESCALAR_STICKER].notna()]
        escalares_cols.append(ESCALAR_STICKER)
    elif args.modo_sticker == "todos":
        escalares_cols += [ESCALAR_STICKER, TEM_STICKER]
    df = df.reset_index(drop=True)
    hold_df = df[df["key"].isin(chaves_hold)].reset_index(drop=True)
    dev_df = df[~df["key"].isin(chaves_hold)].reset_index(drop=True)

    config = {k: v for k, v in vars(args).items() if k not in ("cache_csv", "saida")}
    print(f"  CNN Single-View v2 — {args.folds} folds, até {args.epocas} épocas, device {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"  config: {json.dumps(config)}")
    print(f"  desenvolvimento: {len(dev_df)} imgs ({dev_df['key'].nunique()} animais) | "
          f"teste final: {len(hold_df)} imgs ({hold_df['key'].nunique()} animais) | cache {tamanho_cache}px")
    print(f"  escalares: {escalares_cols}")

    resultados, partes_oof = [], []
    for fold, (treino_df, teste_df) in enumerate(folds_por_animal(dev_df, args.folds, SEED), start=1):
        r = treinar_fold(fold, treino_df, teste_df, escalares_cols, args, tamanho_cache)
        resultados.append(r)
        partes_oof.append(teste_df[["key", "vista", "lote"]].assign(fold=fold, peso_real=r["real"], peso_pred=r["pred"]))
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(partes_oof, ignore_index=True)
    real, pred = oof["peso_real"].to_numpy(), oof["peso_pred"].to_numpy()
    por_fold = [metricas(r["real"], r["pred"]) for r in resultados]
    resumo = resumo_oof(real, pred)

    print(f"\n{'=' * 60}\n  RESULTADO — out-of-fold no desenvolvimento ({args.folds} folds)\n{'=' * 60}")
    for i, m in enumerate(por_fold, start=1):
        print(f"    Fold {i}: MAE {m['mae']:6.2f}kg | R² {m['r2']:.3f} | ±10% {m['acc10']:.1f}% "
              f"| melhor época {resultados[i - 1]['melhor_epoca']}")
    print()
    imprimir_resumo(resumo)

    saida_json = {"config": config, "n_dev": len(dev_df), "n_holdout": len(hold_df),
                  "oof": resumo, "por_fold": por_fold}
    oof.to_csv(os.path.join(args.saida, "predicoes_oof.csv"), index=False)
    np.save(os.path.join(args.saida, "predicoes_oof.npy"), np.array([real, pred]))

    if args.avaliar_holdout and len(hold_df):
        pred_h, real_h = prever_holdout(args, hold_df)
        resumo_h = resumo_oof(real_h, pred_h)
        print(f"{'=' * 60}\n  TESTE FINAL — ensemble de {args.folds} modelos, animais nunca usados\n{'=' * 60}")
        imprimir_resumo(resumo_h)
        saida_json["holdout"] = resumo_h
        hold_df[["key", "vista", "lote"]].assign(peso_real=real_h, peso_pred=pred_h).to_csv(
            os.path.join(args.saida, "predicoes_holdout.csv"), index=False)

    with open(os.path.join(args.saida, "resultados.json"), "w") as f:
        json.dump(saida_json, f, indent=2)

    graficos = os.path.join(args.saida, "graficos")
    os.makedirs(graficos, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(resultados, start=1):
        ax.plot(range(1, len(r["historico"]["val_mae"]) + 1), r["historico"]["val_mae"], label=f"Fold {i}")
    ax.set_xlabel("Época"); ax.set_ylabel("MAE validação interna (kg)")
    ax.set_title("Single-View v2 — validação interna por fold"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(graficos, "sv2_val_mae.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = [min(real.min(), pred.min()) - 10, max(real.max(), pred.max()) + 10]
    ax.scatter(real, pred, s=10, alpha=0.35)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Peso real (kg)"); ax.set_ylabel("Peso predito (kg)")
    ax.set_title(f"Single-View v2 — out-of-fold | R² {resumo['r2']:.3f}"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(graficos, "sv2_dispersao_oof.png"), dpi=150); plt.close(fig)
    print(f"  Modelos, gráficos e resultados em: {args.saida}")


if __name__ == "__main__":
    main()
