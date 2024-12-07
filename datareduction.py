import numpy as np
from time import time
import torch.nn as nn
import torch
import torch.functional as F
import torchvision.transforms as trsf
import os 
from PIL import Image
from tqdm import tqdm

############   Preporcessing ##########

# Le script prend 3-4 min et il fait le preprocessing des images et les enregistre autre part. 
# Ca permet de pas a avoir à le faire par batch pendant l'entrainement du CNN
if __name__ == "__main__":



    input_output_path = "Data/Dashcams_color"
    target_output_path = "Data/Costmapsextended"
# Lister les fichiers dans le dossier et les trier
    files = sorted(os.listdir(input_output_path))  # Tri pour un ordre cohérent

    # Supprimer les fichiers qui ne sont pas retenus
    for i, file in enumerate(files):
        file_path = os.path.join(input_output_path, file)
        if i % 20 != 0:  # Garder une image sur 10
            os.remove(file_path)

    # Lister les fichiers dans le dossier et les trier
    files = sorted(os.listdir(target_output_path))  # Tri pour un ordre cohérent

    # Supprimer les fichiers qui ne sont pas retenus
    for i, file in enumerate(files):
        file_path = os.path.join(target_output_path, file)
        if i % 20 != 0:  # Garder une image sur 10
            os.remove(file_path)