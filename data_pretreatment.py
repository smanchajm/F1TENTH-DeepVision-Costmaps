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

    input_path = "C:\\\\Users\\Robin\\Documents\\POSTECH\\Deep Learning theory\\Projet\\dashcam"
    target_path = "C:\\Users\\Robin\\Documents\\POSTECH\\Deep Learning theory\\Projet\\Costmap"

    input_output_path = "Data/Dashcams_grey"
    target_output_path = "Data/Costmap_grey"

    os.makedirs(input_output_path, exist_ok=True)
    os.makedirs(target_output_path, exist_ok=True)

    input_width = 150
    input_height = 150


    transform = trsf.Compose([
        trsf.Grayscale(num_output_channels=1),# Convertir depuis OpenCV à PIL
     # Convertir en niveaux de gris
        trsf.Resize((input_width, input_height)), # Downsample
        trsf.ToTensor(), 
        # trsf.Normalize(mean=[0.], std=[1.]),# Convertir en tenseur
    ])
    transform_target = trsf.Compose([                  # Convertir depuis OpenCV à PIL
        trsf.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
        trsf.ToTensor(),                   # Convertir en tenseur
    ])

    print(f"[INFO] Début de la conversion des images de la Dashcam")
    dashcam_start = time()

    # Traitement des input 
    for file_name in tqdm(os.listdir(input_path)):
        input_file_path = os.path.join(input_path, file_name)

        # Vérifier si c'est une image
        if os.path.isfile(input_file_path) and file_name.endswith(('.png', '.jpg', '.jpeg')):
            # Charger l'image
            img = Image.open(input_file_path)

            # Appliquer les transformations
            transformed_img = transform(img)

            # Sauvegarder l'image transformée
            output_file_path = os.path.join(input_output_path, file_name)
            transformed_img_pil = trsf.ToPILImage()(transformed_img)
            transformed_img_pil.save(output_file_path)
    dashcam_end = time()    
    print(f"[INFO] Fin de la conversions en {dashcam_end-dashcam_start:.2f} s")


    print(f"[INFO] Début de la conversion des Costmaps")
    costmap_start = time()
    #traitement des targets
    for file_name in tqdm(os.listdir(target_path)):
        target_file_path = os.path.join(target_path, file_name)

        # Vérifier si c'est une image
        if os.path.isfile(target_file_path) and file_name.endswith(('.png', '.jpg', '.jpeg')):
            # Charger la cible
            target = Image.open(target_file_path)

            # Appliquer les transformations
            transformed_target = transform_target(target)

            # Sauvegarder la cible transformée
            output_file_path = os.path.join(target_output_path, file_name)
            transformed_target_pil = trsf.ToPILImage()(transformed_target)
            transformed_target_pil.save(output_file_path)
    costmap_end = time()
    print(f"[INFO] Fin de la conversions en {costmap_end-costmap_start:.2f} s")