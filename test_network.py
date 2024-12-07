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
import cv2



def identity_initialization(layer):
    """
    Applique l'initialisation identité sur une couche nn.Conv2d.
    """
    if isinstance(layer, nn.Conv2d):
        # Remplir les poids de la couche avec 0
        nn.init.constant_(layer.weight, 0)
        
        # Trouver le centre du kernel
        kernel_size = layer.kernel_size[0]  # On suppose kernel_size[0] == kernel_size[1]
        mid = kernel_size // 2  # Index du centre pour les kernels 3x3, c'est 1

        # Initialiser les canaux de sortie en diagonale
        for i in range(min(layer.in_channels, layer.out_channels)):
            layer.weight.data[i, i, mid, mid] = 1  # Assure la diagonale du kernel

        # Initialiser les biais à 0
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class ImageToImageDataset(Dataset):
    def __init__(self, input_folder, target_folder, transform_input=None, transform_target=None,threshold = 30,filter_size = 26):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.input_files = sorted(os.listdir(input_folder))  # Trier pour matcher les paires
        self.target_files = sorted(os.listdir(target_folder))  # Trier pour matcher les paires
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.threshold = threshold
        self.filter_size = filter_size

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Charger l'image d'entrée
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # Charger en niveaux de gris
        
        # Charger l'image cible
        target_path = os.path.join(self.target_folder, self.target_files[idx])
        target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        # Générer le masque avec OpenCV (dilatation)
        _, binary_image = cv2.threshold(target_image, self.threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.filter_size, self.filter_size))  # Kernel circulaire pour une silatation de 10 pixel
        mask = cv2.dilate(binary_image, kernel, iterations=1)  # Appliquer la dilatation
        # mask = (dilated_image == 0).astype('float32')  # Générer le masque binaire (0 = noir, 1 = blanc)

        # Appliquer les transformations si disponibles
        if self.transform_input:
            input_image = self.transform_input(Image.fromarray(input_image))
        if self.transform_target:
            target_image = self.transform_target(target_image)

        return input_image, target_image, torch.from_numpy(mask).unsqueeze(0)
# %%


class ContextNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_factors=[1, 1,1, 2, 4, 8, 16,32, 1],outputchannels=[48,64,128,32,32,32,32,32,1]):
        """
        Implémentation du contexte réseau avec calcul manuel du padding.
        Args:
            in_channels (int): Nombre de canaux en entrée.
            out_channels (int): Nombre de canaux en sortie pour la dernière couche.
            dilation_factors (list): Liste des facteurs de dilation pour chaque couche.
        """
        super(ContextNetwork, self).__init__()
        
        layers = []
        outchannel_previous = in_channels
        for i, tupl in enumerate(zip(dilation_factors,outputchannels)):
            # Calcul manuel du padding
            dilatation,outputchannel = tupl
            kernel_size = 3 if i < len(dilation_factors) - 1 else 1
            padding = (dilatation * (kernel_size - 1)) // 2
            
            layer = nn.Conv2d(
                in_channels=outchannel_previous ,
                out_channels=outputchannel,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilatation
            )
            outchannel_previous = outputchannel
            # Appliquer l'initialisation identité
            identity_initialization(layer)
            layers.append(layer)
            
            if i < len(dilation_factors) - 1:  # Troncature max(·, 0) sauf pour la dernière couche
                layers.append(nn.ReLU(inplace=True))
        
        self.context_module = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.context_module(x)



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ContextNetwork(1,1,outputchannels=[1,1,1,1,1,1,1,1,1])
    model.load_state_dict(torch.load("params1.pth",map_location=torch.device('cpu')))


    BATCH_SIZE = 20

    transform_input = trsf.Compose([
        trsf.ToTensor()
    ])
    transform_output = trsf.Compose([
        trsf.ToPILImage(),
        trsf.ToTensor()
    ])

    dataset = ImageToImageDataset(
        input_folder="Data\\Dashcams",
        target_folder="Data\\Costmaps",
        transform_input=transform_input,
        transform_target=transform_output,
        threshold=150,
        filter_size=50

    )
    data_loader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)


    batch_inputs, batch_targets,masks = next(iter(data_loader))

    out = model(batch_inputs)
    plt.figure(figsize=(10, 5))

    # afficher la première paire
    plt.subplot(2, 2, 1)
    plt.imshow(out[0].squeeze().detach(), cmap='gray')  # input 1
    plt.title("input 1")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(batch_targets[0].squeeze(), cmap='gray')  # target 1
    plt.title("target 1")
    plt.axis('off')

    # afficher la deuxième paire
    plt.subplot(2, 2, 3)
    plt.imshow(masks[0].squeeze().detach())  # input 2
    plt.title("input 2")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(batch_inputs[0].squeeze(), cmap='gray')  # target 2
    plt.title("target 2")
    plt.axis('off')

    plt.tight_layout()
    plt.show()