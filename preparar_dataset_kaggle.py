import os
import numpy as np
import glob
import shutil
from PIL import Image
import sys

COR_BOVINO = {
    "B2": np.array([0, 255, 0, 255]),
    "B3": np.array([255, 30, 249, 255]),
    "B4": np.array([255, 30, 249, 255]),
}

def extrair_peso(filename, lote):
    nome = filename.replace("___fuse.png", "").replace(".jpg", "")
    partes = nome.split("_")
    try:
        indice = 3 if lote == "B4" else 2
        val = float(partes[indice])
        if 50 <= val <= 1000:
            return val
    except:
        pass
    return None

def processar_e_salvar():
    base_dir = os.path.join("data", "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV", "Pixel")
    out_dir = "data_kaggle"
    out_images_dir = os.path.join(out_dir, "images")
    
    if os.path.exists(out_dir):
        print(f"Limpando pasta de saída anterior '{out_dir}'...")
        shutil.rmtree(out_dir)
        
    os.makedirs(out_images_dir, exist_ok=True)
    
    print("Iniciando pré-processamento de imagens (Fundo Preto + Redimensionamento 224x224)...")
    
    pastas_lote = {
        "B2": os.path.join(base_dir, "B2", "Side"),
        "B3": os.path.join(base_dir, "B3"),
        "B4": os.path.join(base_dir, "B4", "Side"),
    }
    
    total_processado = 0
    total_erros = 0
    
    for lote, pasta in pastas_lote.items():
        pasta_imgs = os.path.join(pasta, "images")
        pasta_masks = os.path.join(pasta, "annotations")
        
        if not os.path.exists(pasta_imgs) or not os.path.exists(pasta_masks):
            print(f"Aviso: Pasta para o lote {lote} nao encontrada. Pulando...")
            continue
            
        print(f"\nProcessando lote {lote}...")
        imagens = [f for f in os.listdir(pasta_imgs) if f.endswith(".jpg")]
        
        cor_alvo = COR_BOVINO[lote]
        
        for idx, img_file in enumerate(imagens):
            mask_file = img_file + "___fuse.png"
            mask_path = os.path.join(pasta_masks, mask_file)
            
            if not os.path.exists(mask_path):
                continue
                
            peso = extrair_peso(mask_file, lote)
            if peso is None:
                continue
                
            try:
                img_path = os.path.join(pasta_imgs, img_file)
                img_pil = Image.open(img_path).convert("RGB")
                mask_pil = Image.open(mask_path)
                
                img_np = np.array(img_pil)
                mask_np = np.array(mask_pil)
                
                if mask_np.shape[-1] == 3:
                    alpha = np.full((*mask_np.shape[:2], 1), 255, dtype=np.uint8)
                    mask_np = np.concatenate([mask_np, alpha], axis=-1)
                    
                bovino_mask = np.all(mask_np == cor_alvo, axis=-1)
                
                if bovino_mask.sum() == 0:
                    continue
                    
                masked_img = img_np.copy()
                masked_img[~bovino_mask] = 0 # Define tudo que não for boi como preto (0)
                
                masked_pil = Image.fromarray(masked_img)
                resized_pil = masked_pil.resize((224, 224), Image.Resampling.LANCZOS)
                
                novo_nome = f"{lote}_{int(peso)}_{img_file.replace('.jpg', '')}.png"
                resized_pil.save(os.path.join(out_images_dir, novo_nome))
                
                total_processado += 1
                if total_processado % 500 == 0:
                    print(f"-> {total_processado} imagens processadas com sucesso...")
                    
            except Exception as e:
                total_erros += 1
                
    print("\n" + "="*80)
    print("RESUMO DO PRE-PROCESSAMENTO:")
    print(f"Imagens processadas e salvas com sucesso: {total_processado}")
    print(f"Erros encontrados: {total_erros}")
    print(f"Pasta de saida criada: '{out_dir}'")
    print("="*80 + "\n")
    
    if total_processado > 0:
        total_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(out_images_dir, "*")))
        print(f"Tamanho total do novo dataset: {total_size / (1024*1024):.2f} MB")
        return True
    else:
        print("Nenhuma imagem foi processada. Verifique os caminhos.")
        return False

if __name__ == '__main__':
    processar_e_salvar()
