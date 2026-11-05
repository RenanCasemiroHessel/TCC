"""
Peças compartilhadas do pipeline single-view v2: recorte, features escalares,
modelo e transforms. Usado pelo cache, pelo treino e pela inferência — o
pré-processamento precisa ser idêntico nos três, então mora num lugar só.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# cores das máscaras por lote (RGB), conforme Readme.md do dataset
CORES = {
    "B2": {"cattle": (0, 255, 0), "sticker": (255, 240, 0)},
    "B3": {"cattle": (255, 30, 249), "sticker": (0, 117, 255)},
    "B4": {"cattle": (255, 30, 249), "sticker": (0, 117, 255)},
}

TAMANHO_CACHE = 256
MARGEM = 0.08
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ESCALARES_BASE = ["log_area_norm", "log_altura_rel", "log_largura_rel", "sexo_M"]
ESCALAR_STICKER = "log_sticker_ratio"
TEM_STICKER = "tem_sticker"   # 1 se o sticker aparece na foto; senão o ratio é preenchido e isto fica 0

# construtor, pesos ImageNet e dimensão do vetor de features de cada backbone
BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, 768),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT, 1280),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT, 768),
    "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.DEFAULT, 1280),
}


def sexo_do_arquivo(path):
    """Último campo do nome: ..._F.jpg / ..._M.jpg -> 0.0 / 1.0 (None se não reconhecer)."""
    ultimo = os.path.basename(path).rsplit(".", 1)[0].split("_")[-1]
    return {"F": 0.0, "M": 1.0}.get(ultimo)


def mascaras(mask_rgb, lote):
    cattle = np.all(mask_rgb == np.array(CORES[lote]["cattle"]), axis=-1)
    sticker = np.all(mask_rgb == np.array(CORES[lote]["sticker"]), axis=-1)
    return cattle, sticker


def remover_sticker(img_rgb, cattle_mask, sticker_mask):
    """Preenche o sticker com a textura do pelo ao redor e o devolve como parte do boi.

    Sem isso o recorte fica com um buraco preto no corpo — quadrado no B2, redondo
    no B3/B4 (atalho pro lote) e inexistente em foto real sem sticker.
    """
    if sticker_mask.sum() == 0:
        return img_rgb, cattle_mask
    alvo = cv2.dilate(sticker_mask.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    return cv2.inpaint(img_rgb, alvo, 7, cv2.INPAINT_TELEA), cattle_mask | sticker_mask


def escalares_mascara(cattle_mask):
    """Tamanho do boi relativo ao quadro — o recorte joga fora a escala, isto devolve."""
    H, W = cattle_mask.shape
    rows = np.where(cattle_mask.any(axis=1))[0]
    cols = np.where(cattle_mask.any(axis=0))[0]
    altura = rows.max() - rows.min() + 1
    largura = cols.max() - cols.min() + 1
    return {
        "log_area_norm": float(np.log(cattle_mask.sum() / (H * W))),
        "log_altura_rel": float(np.log(altura / H)),
        "log_largura_rel": float(np.log(largura / W)),
        # forma da silhueta — usadas pelos modelos tabulares
        "fill_ratio": float(cattle_mask.sum() / (altura * largura)),
        "proporcao": float(largura / altura),
    }


def recortar_boi(img_rgb, cattle_mask, tamanho=TAMANHO_CACHE, margem=MARGEM):
    """Fundo preto, recorte quadrado centrado no boi mantendo a proporção (sem achatar)."""
    out = img_rgb.copy()
    out[~cattle_mask] = 0
    rows = np.where(cattle_mask.any(axis=1))[0]
    cols = np.where(cattle_mask.any(axis=0))[0]
    y0, y1 = rows.min(), rows.max() + 1
    x0, x1 = cols.min(), cols.max() + 1
    lado = int(max(y1 - y0, x1 - x0) * (1 + 2 * margem))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    left, top = cx - lado // 2, cy - lado // 2
    crop = Image.fromarray(out).crop((left, top, left + lado, top + lado))  # fora da imagem vira preto
    return crop.resize((tamanho, tamanho), Image.LANCZOS)


def transform_treino(img_size=IMG_SIZE, forte=False):
    if not forte:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.06, 0.06), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.03),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
    ])


def transform_avaliacao(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def construir_backbone(nome, pretrained=True):
    """Backbone sem a camada de classificação, devolvendo um vetor por imagem."""
    construtor, pesos, dim = BACKBONES[nome]
    base = construtor(weights=pesos if pretrained else None)
    if nome.startswith("resnet"):
        corpo = nn.Sequential(*list(base.children())[:-1])
    elif nome.startswith("convnext"):
        corpo = nn.Sequential(base.features, base.avgpool, base.classifier[0])  # [0] = LayerNorm2d
    else:  # efficientnet
        corpo = nn.Sequential(base.features, base.avgpool)
    return corpo, dim


class SingleViewNet(nn.Module):
    """Backbone + escalares concatenados antes da cabeça. Saída linear (log-peso padronizado).

    Com os padrões (resnet18, 256, 0.4) a estrutura é a mesma dos checkpoints já
    treinados, então eles continuam carregando.
    """

    def __init__(self, n_escalares, backbone="resnet18", cabeca=256, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone, dim = construir_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Linear(dim + n_escalares, cabeca), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(cabeca, 1),
        )

    def forward(self, x, escalares):
        feat = self.backbone(x).flatten(1)
        return self.head(torch.cat([feat, escalares], dim=1))


def metricas(real, pred):
    erro = np.abs(pred - real)
    return {
        "mae": float(erro.mean()),
        "rmse": float(np.sqrt(((pred - real) ** 2).mean())),
        "r2": float(1 - ((pred - real) ** 2).sum() / ((real - real.mean()) ** 2).sum()),
        "pearson_r": float(np.corrcoef(real, pred)[0, 1]),
        "mape": float((erro / real).mean() * 100),
        "acc10": float((erro / real <= 0.10).mean() * 100),
    }


CORTES_FAIXAS = (130.0, 190.0)   # mesmos cortes do F1 macro do TCC1: leve / médio / pesado
LIMIAR_PESADO = 230.0


def nomes_faixas(cortes):
    nomes = [f"<{cortes[0]:.0f}"]
    nomes += [f"{a:.0f}-{b:.0f}" for a, b in zip(cortes[:-1], cortes[1:])]
    return nomes + [f">={cortes[-1]:.0f}"]


def metricas_classificacao(real, pred, cortes=CORTES_FAIXAS, limiar=LIMIAR_PESADO):
    """Converte peso real e predito em faixas e mede como classificação.

    Regressão não tem F1 nativo — o resultado depende dos cortes. Por isso vêm
    F1 macro e acurácia balanceada (não inflam com a classe majoritária, o erro
    do "97% de acurácia" do TCC1) e a detecção binária de pesados, que é a
    pergunta mais próxima de uma decisão de venda.
    """
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    yt, yp = np.digitize(real, cortes), np.digitize(pred, cortes)
    rotulos = list(range(len(cortes) + 1))
    p, r, f, n = precision_recall_fscore_support(yt, yp, labels=rotulos, zero_division=0)
    pesado_real, pesado_pred = real >= limiar, pred >= limiar
    pb, rb, fb, _ = precision_recall_fscore_support(pesado_real, pesado_pred, average="binary", zero_division=0)
    return {
        "cortes_kg": list(cortes),
        "por_faixa": {nome: {"precisao": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": int(n[i])}
                      for i, nome in enumerate(nomes_faixas(cortes))},
        "f1_macro": float(f1_score(yt, yp, labels=rotulos, average="macro", zero_division=0)),
        "f1_ponderado": float(f1_score(yt, yp, labels=rotulos, average="weighted", zero_division=0)),
        "acuracia_balanceada": float(balanced_accuracy_score(yt, yp)),
        "matriz_confusao": confusion_matrix(yt, yp, labels=rotulos).tolist(),
        "pesados": {"limiar_kg": limiar, "precisao": float(pb), "recall": float(rb), "f1": float(fb),
                    "n_reais": int(pesado_real.sum()), "n_preditos": int(pesado_pred.sum())},
    }


def imprimir_classificacao(c):
    nomes = list(c["por_faixa"])
    print(f"  Classificação por faixa (cortes {'/'.join(f'{x:.0f}' for x in c['cortes_kg'])} kg)")
    print(f"    {'faixa (kg)':>10s} {'precisão':>9s} {'recall':>7s} {'F1':>6s} {'n':>6s}")
    for nome in nomes:
        m = c["por_faixa"][nome]
        print(f"    {nome:>10s} {m['precisao']:9.3f} {m['recall']:7.3f} {m['f1']:6.3f} {m['n']:6d}")
    print(f"    F1 macro {c['f1_macro']:.3f} | F1 ponderado {c['f1_ponderado']:.3f} "
          f"| acurácia balanceada {c['acuracia_balanceada']:.3f}")
    print(f"    Matriz de confusão (linha = real, coluna = predito, ordem {' / '.join(nomes)}):")
    for nome, linha in zip(nomes, c["matriz_confusao"]):
        print(f"      {nome:>8s} " + " ".join(f"{v:6d}" for v in linha))
    pz = c["pesados"]
    print(f"  Detecção de pesados (>= {pz['limiar_kg']:.0f} kg): precisão {pz['precisao']:.3f} | "
          f"recall {pz['recall']:.3f} | F1 {pz['f1']:.3f} "
          f"({pz['n_reais']} pesados reais, {pz['n_preditos']} apontados pelo modelo)")


def resumo_oof(real, pred):
    """Métricas out-of-fold + IC95% bootstrap do MAE + cobertura de erro + viés nos extremos."""
    erro = np.abs(pred - real)
    erro_rel = erro / real * 100
    leves, pesados = real < np.percentile(real, 10), real > np.percentile(real, 90)
    rng = np.random.default_rng(0)
    boots = [erro[rng.integers(0, len(erro), len(erro))].mean() for _ in range(2000)]
    return {
        **metricas(real, pred),
        "mae_ic95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "erro_p80_kg": float(np.percentile(erro, 80)), "erro_p90_kg": float(np.percentile(erro, 90)),
        "erro_p80_pct": float(np.percentile(erro_rel, 80)), "erro_p90_pct": float(np.percentile(erro_rel, 90)),
        "vies_10pct_mais_leves_kg": float((pred - real)[leves].mean()),
        "vies_10pct_mais_pesados_kg": float((pred - real)[pesados].mean()),
        "faixa_predita": [float(pred.min()), float(pred.max())],
        "faixa_real": [float(real.min()), float(real.max())],
        "classificacao": metricas_classificacao(real, pred),
    }


def imprimir_resumo(r):
    print(f"""  MAE       : {r['mae']:.2f} kg  (IC95% bootstrap {r['mae_ic95'][0]:.2f}-{r['mae_ic95'][1]:.2f})
  RMSE      : {r['rmse']:.2f} kg
  R²        : {r['r2']:.4f}
  Pearson r : {r['pearson_r']:.4f}
  MAPE      : {r['mape']:.1f} %
  Acc ±10%  : {r['acc10']:.1f} %
  80% das predições erram até ±{r['erro_p80_kg']:.1f}kg (±{r['erro_p80_pct']:.1f}%)
  Viés nos 10% mais leves: {r['vies_10pct_mais_leves_kg']:+.1f}kg | mais pesados: {r['vies_10pct_mais_pesados_kg']:+.1f}kg
  Faixa predita {r['faixa_predita'][0]:.0f}-{r['faixa_predita'][1]:.0f}kg vs real {r['faixa_real'][0]:.0f}-{r['faixa_real'][1]:.0f}kg
""")
    imprimir_classificacao(r["classificacao"])
    print()


def folds_por_animal(df, n_folds, seed=42):
    """Mesma divisão por animal pra CNN e tabulares (mesmo CSV + mesma seed = mesmos folds)."""
    from sklearn.model_selection import KFold
    chaves = df["key"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i_tr, i_te in kf.split(chaves):
        yield (df[df["key"].isin(chaves[i_tr])].reset_index(drop=True),
               df[df["key"].isin(chaves[i_te])].reset_index(drop=True))


def prever_tta(model, imgs, escalares):
    """Média da predição original e espelhada (flip horizontal), na escala do alvo."""
    return (model(imgs, escalares) + model(torch.flip(imgs, dims=[3]), escalares)) / 2
