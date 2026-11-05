"""
Utilitários de salvamento seguro (sem sobrescrever arquivos existentes).
"""

import os
import json

def safe_path(path):
    """Retorna caminho versionado se o original já existir. Ex: arquivo_v2.npy"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    v = 2
    while os.path.exists(f"{base}_v{v}{ext}"):
        v += 1
    return f"{base}_v{v}{ext}"

def latest_file(base_path):
    """Retorna o arquivo mais recente (original ou última versão _vN)."""
    if not os.path.exists(base_path):
        base, ext = os.path.splitext(base_path)
        v = 2
        while os.path.exists(f"{base}_v{v}{ext}"):
            v += 1
        last = f"{base}_v{v-1}{ext}"
        return last if os.path.exists(last) else None
    base, ext = os.path.splitext(base_path)
    latest = base_path
    v = 2
    while os.path.exists(f"{base}_v{v}{ext}"):
        latest = f"{base}_v{v}{ext}"
        v += 1
    return latest

def salvar_json(results_path, chave_base, dados):
    """
    Salva dados no JSON sem sobrescrever chaves existentes.
    Se a chave já existe, usa 'chave v2', 'chave v3', etc.
    Retorna a chave efetivamente usada.
    """
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    chave = chave_base
    v = 2
    while chave in existing:
        chave = f"{chave_base} v{v}"
        v += 1
    existing[chave] = dados
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    if chave != chave_base:
        print(f"  [!] Chave '{chave_base}' já existia → salvo como '{chave}'")
    return chave
