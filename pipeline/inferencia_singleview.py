"""
Inferência single-view v2: ensemble dos modelos dos K folds + TTA (flip).

Cada fold foi treinado em 80% dos animais; a média dos K costuma errar menos
que qualquer um sozinho. A média é feita no espaço do alvo (log-peso).

Recebe a imagem e a máscara do boi já prontas — de onde vem a máscara
(anotação do dataset ou YOLO no app) fica a cargo de quem chama.

Uso pela linha de comando (com uma imagem e máscara do dataset):
  python inferencia_singleview.py --imagem foto.jpg --mascara foto.jpg___fuse.png --lote B4 --sexo F
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageOps

from caminhos import MODELOS
from pipeline_singleview import (
    ESCALAR_STICKER, TEM_STICKER, SingleViewNet, escalares_mascara, mascaras, prever_tta,
    recortar_boi, remover_sticker, transform_avaliacao,
)


class EnsembleSingleView:
    def __init__(self, pasta_modelos=os.path.join(MODELOS, "modelos_final_singleview"), device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = sorted(p for p in glob.glob(os.path.join(pasta_modelos, "*sv2_fold*.pth"))
                       if not p.endswith("_latest.pth"))
        if not paths:
            raise FileNotFoundError(f"Nenhum modelo *sv2_fold*.pth em {pasta_modelos}")
        self.membros = []
        for p in paths:
            ck = torch.load(p, map_location=self.device, weights_only=False)
            # checkpoints antigos não guardavam a arquitetura: eram todos resnet18 / 256 / 0.4 / 224px
            model = SingleViewNet(len(ck["escalares"]), ck.get("backbone", "resnet18"), ck.get("cabeca", 256),
                                  ck.get("dropout", 0.4), pretrained=False).to(self.device)
            model.load_state_dict(ck["model_state_dict"])
            model.eval()
            self.membros.append((model, ck))
        ck0 = self.membros[0][1]
        self.escalares = ck0["escalares"]
        self.tamanho_cache = ck0.get("tamanho_cache", 256)
        self.tf = transform_avaliacao(ck0.get("img_size", 224))

    @torch.no_grad()
    def prever(self, img_rgb, cattle_mask, sexo_m, sticker_mask=None):
        """img_rgb: HxWx3 uint8 | cattle_mask: HxW bool | sexo_m: 0.0 (F) ou 1.0 (M)."""
        tem_sticker = sticker_mask is not None and sticker_mask.sum() >= 50
        if sticker_mask is not None:
            img_rgb, cattle_mask = remover_sticker(img_rgb, cattle_mask, sticker_mask)
        brutos = escalares_mascara(cattle_mask)
        brutos["sexo_M"] = float(sexo_m)
        ratio = float(np.log(cattle_mask.sum() / sticker_mask.sum())) if tem_sticker else None
        if ESCALAR_STICKER in self.escalares and TEM_STICKER not in self.escalares and not tem_sticker:
            raise ValueError("Modelo treinado só com fotos com sticker, mas o sticker não foi encontrado na foto.")

        img = self.tf(recortar_boi(img_rgb, cattle_mask, self.tamanho_cache)).unsqueeze(0).to(self.device)

        logs = []
        for model, ck in self.membros:
            if TEM_STICKER in self.escalares:
                brutos[TEM_STICKER] = 1.0 if tem_sticker else 0.0
                brutos[ESCALAR_STICKER] = ratio if tem_sticker else ck["sticker_preenchimento"]
            elif ESCALAR_STICKER in self.escalares:
                brutos[ESCALAR_STICKER] = ratio
            valores = np.array([brutos[c] for c in self.escalares], dtype=np.float32)
            esc = torch.from_numpy((valores - ck["esc_mean"]) / ck["esc_std"]).float().unsqueeze(0).to(self.device)
            logs.append(float(prever_tta(model, img, esc).item()) * ck["sd"] + ck["mu"])
        por_modelo = np.exp(logs)
        return {
            "peso_kg": float(np.exp(np.mean(logs))),
            "por_modelo_kg": por_modelo.round(1).tolist(),
            "desvio_entre_modelos_kg": float(por_modelo.std()),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--mascara", required=True, help="máscara anotada do dataset (*___fuse.png)")
    ap.add_argument("--lote", required=True, choices=["B2", "B3", "B4"])
    ap.add_argument("--sexo", required=True, choices=["F", "M"])
    ap.add_argument("--modelos", default=os.path.join(MODELOS, "modelos_final_singleview"))
    args = ap.parse_args()

    img = np.array(ImageOps.exif_transpose(Image.open(args.imagem)).convert("RGB"))
    cattle, sticker = mascaras(np.array(Image.open(args.mascara).convert("RGB")), args.lote)
    ens = EnsembleSingleView(args.modelos)
    r = ens.prever(img, cattle, 1.0 if args.sexo == "M" else 0.0, sticker)
    print(f"  Peso estimado: {r['peso_kg']:.1f} kg  (ensemble de {len(r['por_modelo_kg'])} modelos)")
    print(f"  Por modelo   : {r['por_modelo_kg']}  | desvio {r['desvio_entre_modelos_kg']:.1f} kg")


if __name__ == "__main__":
    main()
