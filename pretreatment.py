import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Dataset personnalisé
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
        input_image = cv2.imread(input_path)  # Charger avec OpenCV
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convertir en RGB

        # Charger l'image cible
        target_path = os.path.join(self.target_folder, self.target_files[idx])
        target_image = cv2.imread(target_path)  # Charger avec OpenCV
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)  # Convertir en RGB

        # Appliquer les transformations
        if self.transform_input:
            input_image = self.transform_input(input_image)
        if self.transform_target:
            target_image = self.transform_target(target_image)

        return input_image, target_image


# Dossiers d'entrée et de sortie
input_folder = "C:\Users\Robin\Documents\POSTECH\Deep Learning theory\Projet\dashcam"  # Remplace par le chemin de tes inputs
target_folder = "C:\Users\Robin\Documents\POSTECH\Deep Learning theory\Projet\costmap"  # Remplace par le chemin des cibles

# Transformations pour l'entrée et la cible
transform = transforms.Compose([
    transforms.ToPILImage(),                  # Convertir depuis OpenCV à PIL
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
    transforms.Resize((720 // 8, 1280 // 8)), # Downsample
    transforms.ToTensor(),                   # Convertir en tenseur
])

# Créer le dataset
dataset = ImageToImageDataset(
    input_folder=input_folder,
    target_folder=target_folder,
    transform_input=transform,
    transform_target=transform
)

# DataLoader
batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Exemple : Afficher deux paires d'images (input et target)
batch_inputs, batch_targets = next(iter(data_loader))

plt.figure(figsize=(10, 5))

# Afficher la première paire
plt.subplot(2, 2, 1)
plt.imshow(batch_inputs[0].squeeze(), cmap='gray')  # Input 1
plt.title("Input 1")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(batch_targets[0].squeeze(), cmap='gray')  # Target 1
plt.title("Target 1")
plt.axis('off')

# Afficher la deuxième paire
plt.subplot(2, 2, 3)
plt.imshow(batch_inputs[1].squeeze(), cmap='gray')  # Input 2
plt.title("Input 2")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(batch_targets[1].squeeze(), cmap='gray')  # Target 2
plt.title("Target 2")
plt.axis('off')

plt.tight_layout()
plt.show()
