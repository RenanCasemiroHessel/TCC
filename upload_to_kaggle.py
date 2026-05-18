import os
import json
import sys
import time

def check_and_authenticate():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Instancia e autentica a API do Kaggle
        api = KaggleApi()
        api.authenticate()
        
        # Recupera o username configurado automaticamente pelo cliente
        username = api.config_values.get('username')
        if username:
            return api, username
            
        # Caso alternativo se o username nao estiver em config_values
        # Tentamos ler diretamente do kaggle.json caso ele exista
        kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_json):
            with open(kaggle_json, 'r', encoding='utf-8') as f:
                creds = json.load(f)
                return api, creds.get("username")
                
        # Se nao achou o username mas autenticou, usa um default/detectado
        return api, "renanhessel"
        
    except Exception as e:
        print("\n" + "="*80)
        print("[ERRO] Nao foi possivel autenticar com o Kaggle.")
        print("Certifique-se de que:")
        print("1. O token do Kaggle (access_token ou kaggle.json) esta na pasta C:\\Users\\renan\\.kaggle\\")
        print(f"Detalhes do erro: {e}")
        print("="*80 + "\n")
        return None, None

def main():
    print("Iniciando processo de autenticação com o Kaggle...")
    api, username = check_and_authenticate()
    if not api or not username:
        sys.exit(1)
        
    print(f"Usuario autenticado com sucesso: {username}")
    
    # Determina dinamicamente qual pasta enviar (prioriza a leve pré-processada)
    if os.path.exists("data_kaggle"):
        upload_dir = "data_kaggle"
        dataset_slug = "bmgf-livestock-weight-cv-preprocessed"
        dataset_title = "BMGF Livestock Weight CV - Preprocessed"
        print(f"-> Detectado dataset pré-processado leve: '{upload_dir}'")
    elif os.path.exists("data"):
        upload_dir = "data"
        dataset_slug = "bmgf-livestock-weight-cv-raw"
        dataset_title = "BMGF Livestock Weight CV - Raw"
        print(f"-> Detectado dataset bruto completo: '{upload_dir}'")
    else:
        print("[ERRO] Nenhuma pasta de dados ('data_kaggle' ou 'data') foi encontrada no diretório atual.")
        sys.exit(1)
        
    # Inicializando metadados do Dataset no formato correto do Kaggle
    metadata_path = os.path.join(upload_dir, "dataset-metadata.json")
    
    metadata = {
        "title": dataset_title,
        "id": f"{username}/{dataset_slug}",
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }
    
    # Salvando dataset-metadata.json dentro da pasta
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Arquivo de metadados gerado com sucesso em: {metadata_path}")
    
    print("\n" + "="*80)
    print(f"INICIANDO O COMPACTAMENTO E UPLOAD DO DATASET: {dataset_title}")
    print("Por favor, deixe o terminal aberto!")
    print("="*80 + "\n")
    
    # Iniciando o upload
    try:
        # Envia o dataset como PRIVADO (public=False)
        print(f"Enviando dados da pasta '{upload_dir}' para os servidores do Kaggle...")
        start_time = time.time()
        api.dataset_create_new(
            folder=upload_dir,
            dir_mode="zip",
            quiet=False,
            public=False
        )
        end_time = time.time()
        
        duration = (end_time - start_time) / 60
        print(f"\nUPLOAD FINALIZADO COM SUCESSO EM {duration:.2f} MINUTOS!")
        print(f"Voce pode acessar seu dataset privado em: https://www.kaggle.com/datasets/{username}/{dataset_slug}")
        
    except Exception as e:
        print(f"\n[ERRO] durante o upload: {e}")
        print("Verifique sua conexao e se ha espaco em disco disponivel no seu computador.")
        sys.exit(1)

if __name__ == '__main__':
    main()
