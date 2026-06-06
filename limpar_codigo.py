import os
import re

FILES = [
    "train_linear_regression.py", "train_xgboost.py", "train_xgboost_multiview.py",
    "train_lr_multiview.py", "train_mlp.py", "train_mlp_multiview.py",
    "treinar_cnn.py", "treinar_cnn_multiview.py", "comparar_modelos.py",
    "predizer_imagem.py", "filtrar_dataset.py", "utils_tcc.py",
    "visualizar_segmentacao.py", "gerar_graficos_tcc.py", "tune_xgboost.py",
    "preparar_dataset_kaggle.py",
]

BASE = "/home/gustavo/Imagens/Tccgados/TCC"

PALAVRAS_KEEP = [
    'erro', 'ERRO', 'aviso', 'AVISO', 'salvo', 'poca', 'Loss', 'Melhor',
    'checkpoint', 'SUCESSO', 'dataset', 'not found', 'encontrado',
]

for fname in FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        continue

    with open(fpath) as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        stripped = line.strip()

        if re.search(r'print\(.*["\']={8,}', line):
            continue
        if re.search(r'print\(f.*\*\s*60', line) and "=" in line:
            continue

        line = re.sub(r'\[(\d+)/(\d+)\]\s+', '', line)

        if re.search(r'print\(f?["\'][ ]{5,}', line):
            if not any(kw in line for kw in PALAVRAS_KEEP):
                continue

        if re.match(r'^\s*#\s*(Carregar|Calcular|Plotar|Gerar|Salvar|Criar|Definir|Configurar|Mostrar|Exibir|Verificar)\b', line, re.IGNORECASE):
            continue

        new_lines.append(line)

    final = []
    prev_blank = False
    for line in new_lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        final.append(line)
        prev_blank = is_blank

    with open(fpath, 'w') as f:
        f.writelines(final)

    print(f"  ok: {fname}")
