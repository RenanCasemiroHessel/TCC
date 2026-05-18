import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Configuração de dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CattleWeightDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.data = []
        
        print("Mapeando pesos a partir dos nomes dos arquivos...")
        for path in image_paths:
            basename = os.path.basename(path)
            # Exemplo: 1.0_r_117_10.0_F.jpg___fuse.png -> peso = 117
            try:
                parts = basename.split('_')
                # Procura a terceira parte que geralmente é o peso, ou ajusta conforme necessidade
                # Pode haver variação entre B2/B3/B4. 
                # B2/B3: id_vista_peso_sufixo. B4: id_batch-sub_vista_peso_sexo_sufixo
                # Nos arquivos listados, o formato era X.X_r_PESO_...
                peso_str = parts[2]
                peso = float(peso_str)
                if 50 <= peso <= 1000: # Filtro de sanidade
                    self.data.append((path, peso))
            except Exception as e:
                pass
                
        print(f"Total de imagens carregáveis com pesos válidos: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, weight = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(weight, dtype=torch.float32)

def main():
    data_dir = os.path.join("data", "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV")
    
    # Busca todas as imagens recursivamente (neste caso os fuses/images)
    print("Buscando imagens nas pastas...")
    all_images = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True) + \
                 glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
                 
    # Para o teste não demorar horas na CPU, vamos limitar a uma amostragem aleatória se necessário
    # Para o treinamento completo, utilizamos todas as imagens encontradas.
    # np.random.shuffle(all_images)
    # all_images = all_images[:1000] 

    
    if len(all_images) == 0:
        print("Nenhuma imagem encontrada na pasta data.")
        return
        
    # Transformações necessárias para a ResNet (Redimensionamento para 128x128 para poupar RAM/Tempo)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CattleWeightDataset(all_images, transform=transform)
    
    if len(dataset) == 0:
         print("Nenhum peso válido extraído. Verifique o formato dos arquivos.")
         return
         
    # Extrair listas para split
    img_paths = [x[0] for x in dataset.data]
    weights = [x[1] for x in dataset.data]
    
    # Split
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(img_paths, weights, test_size=0.2, random_state=42)
    
    train_dataset = CattleWeightDataset(X_train_paths, transform=transform)
    test_dataset = CattleWeightDataset(X_test_paths, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\nTreinando com {len(train_dataset)} imagens, testando com {len(test_dataset)} imagens.")
    print(f"Usando processador/placa de vídeo: {device}")
    
    # Construção do Modelo (Transfer Learning)
    model = models.resnet18(pretrained=True)
    # Substituir a última camada que prevê 1000 classes por 1 neurônio (Regressão Linear do peso)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 15 # Treinamento mais longo para buscar convergência
    
    print(f"Iniciando treinamento definitivo por {num_epochs} épocas...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1) # shape [batch, 1]
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Época {epoch+1}/{num_epochs} - MSE Loss: {epoch_loss:.2f}")
        
    end_time = time.time()
    print(f"\nTreinamento concluído em {(end_time - start_time):.2f} segundos.")
    
    print("\nAvaliando modelo no Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
            
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    
    print("\n--- RESULTADOS DA CNN (ResNet-18) ---")
    print(f"MAE: {mae:.2f} kg")
    print(f"RMSE: {rmse:.2f} kg")
    print(f"R^2: {r2:.4f}")
    
    # Salvar o modelo
    torch.save(model.state_dict(), "resnet18_pesogado.pth")
    print("Modelo salvo como 'resnet18_pesogado.pth'")
    
    # Atualizar o arquivo de resultados automaticamente
    with open("resultados_experimentos.md", "a", encoding="utf-8") as f:
        f.write("\n\n## 7. Deep Learning - CNN ResNet-18 (Treinamento Completo)\n")
        f.write("**Data/Hora:** Automático (Fim do script `train_cnn.py`)\n")
        f.write(f"**Arquitetura:** ResNet-18\n**Entrada:** Imagens reais 128x128 px\n")
        f.write(f"**Condições do Treino:** Todas as imagens válidas ({len(train_dataset)} treino / {len(test_dataset)} teste) por {num_epochs} Épocas.\n\n")
        f.write("### Métricas de Desempenho (Generalização)\n")
        f.write("| Base | MAE (kg) | RMSE (kg) | R² |\n")
        f.write("|------|----------|-----------|----|\n")
        f.write(f"| **Teste** | {mae:.2f} | {rmse:.2f} | {r2:.4f} |\n")
        f.write("\n*(Registro gerado automaticamente no fim do treinamento)*\n")

if __name__ == '__main__':
    main()
