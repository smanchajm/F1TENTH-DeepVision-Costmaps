# %%
import numpy as np
from time import time
import torch.nn as nn
import torch
import torch.functional as F
import torchvision.transforms as trsf
from torch.utils.data import DataLoader,Dataset
import os 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import torch.optim as optim
import wandb




class ImageToImageDataset(Dataset):
    def __init__(self, input_folder, target_folder, transform_input=None, transform_target=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.input_files = sorted(os.listdir(input_folder))  # Trier pour matcher les paires
        self.target_files = sorted(os.listdir(target_folder))  # Trier pour matcher les paires
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Charger l'image d'entrée
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        image = Image.open(input_path)
        # Charger l'image cible
        target_path = os.path.join(self.target_folder, self.target_files[idx])
        target = Image.open(target_path)

        # Appliquer les transformations
        if self.transform_input:
            input_image = self.transform_input(image)
        if self.transform_target:
            target_image = self.transform_target(target)

        return input_image, target_image



class UNet150(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,complexity_multiplier = 1):
        super(UNet150, self).__init__()
        self.cm = complexity_multiplier
        # Contraction (Encoder)
        self.encoder1 = self.double_conv(in_channels, self.cm*16)
        self.pool1 = nn.MaxPool2d(2)  # 150 -> 75

        self.encoder2 = self.double_conv(self.cm*16, self.cm*32)
        self.pool2 = nn.MaxPool2d(2)  # 75 -> 37

        # Bottleneck
        self.bottleneck = self.double_conv(self.cm*32, self.cm*64)

        # Expansion (Decoder)
        self.upconv2 = nn.ConvTranspose2d(self.cm*64, self.cm*32, kernel_size=2, stride=2, output_padding=1)  # 37 -> 75
        self.decoder2 = self.double_conv(self.cm*64, self.cm*32)

        self.upconv1 = nn.ConvTranspose2d(self.cm*32, self.cm*16, kernel_size=2, stride=2, output_padding=0)  # 75 -> 150
        self.decoder1 = self.double_conv(self.cm*32, self.cm*16)

        # Output layer
        self.final_conv = nn.Conv2d(self.cm*16, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """Two consecutive convolution layers with ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # 150x150 -> 32 channels
        pool1 = self.pool1(enc1)  # 150 -> 75

        enc2 = self.encoder2(pool1)  # 75x75 -> 64 channels
        pool2 = self.pool2(enc2)  # 75 -> 37

        # Bottleneck
        bottleneck = self.bottleneck(pool2)  # 37x37 -> 256 channels

        # Decoder
        up2 = self.upconv2(bottleneck)  # 37 -> 75
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))  # Skip connection

        up1 = self.upconv1(dec2)  # 75 -> 150
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))  # Skip connection

        # Output layer
        output = self.final_conv(dec1)  # 150x150 -> out_channels
        return output


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project="F1TENTH"
    )

    BATCH_SIZE = 5

    transform_input = trsf.Compose([
        trsf.ToTensor()
    ])
    transform_output = trsf.Compose([
        trsf.ToTensor()
    ])

    dataset = ImageToImageDataset(
        input_folder="Data\\Dashcams",
        target_folder="Data\\Costmaps",
        transform_input=transform_input,
        transform_target=transform_output

    )
    data_loader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)

    
    # batch_inputs, batch_targets = next(iter(data_loader))

    # plt.figure(figsize=(10, 5))

    # # afficher la première paire
    # plt.subplot(2, 2, 1)
    # plt.imshow(batch_inputs[0].squeeze(), cmap='gray')  # input 1
    # plt.title("input 1")
    # plt.axis('off')

    # plt.subplot(2, 2, 2)
    # plt.imshow(batch_targets[0].squeeze(), cmap='gray')  # target 1
    # plt.title("target 1")
    # plt.axis('off')

    # # afficher la deuxième paire
    # plt.subplot(2, 2, 3)
    # plt.imshow(batch_inputs[1].squeeze(), cmap='gray')  # input 2
    # plt.title("input 2")
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.imshow(batch_targets[1].squeeze(), cmap='gray')  # target 2
    # plt.title("target 2")
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    
    # Définir la loss (L1 Loss) et l'optimiseur
    model = UNet150(1,1,complexity_multiplier=4)
    model.to(device)
    criterion = nn.L1Loss()  # L1 Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimiseur Adam avec un taux d'apprentissage de 0.001

    # Entraînement
    num_epochs = 1
    model.train()
    

    for epoch in range(num_epochs):
        print(f"[INFO] Début de l'époche {epoch}")
        running_loss = 0.0
        for batch_inputs, batch_targets in tqdm(data_loader):
            # Mettre à jour les gradients à 0
            optimizer.zero_grad()
            
            # Passer les données dans le modèle
            s1 = time()
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs)
            e1 = time()
            # print(f"temps de la forward pass {e1-s1:.6f}")
            
            
            # Calculer la perte
            i2 = time()
            loss = criterion(outputs, batch_targets)
            i1 = time() 
            # Backpropagation
            loss.backward()
            s2 = time()
            # Optimisation
            optimizer.step()
            e2 = time()
            # print(f"temps du calcul de la loss {i1-i2:.6f}")
            # print(f"temps du optimizer step {e2-s2:.6f}")
            # print(f"temps du backward {s2-i1:.6f}")
            # Accumuler la perte
            running_loss += loss.item()
            wandb.log({"loss":loss.item()})
        
        # Afficher la perte moyenne par époque
        print(f"Époque [{epoch + 1}/{num_epochs}], Perte moyenne : {running_loss / len(data_loader):.4f}")

    print("Entraînement terminé !")



# %%
