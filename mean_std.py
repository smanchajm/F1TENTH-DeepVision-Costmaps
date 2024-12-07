import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import cv2
import torchvision
import torch


if __name__ == "__main__":
    dir_path = "Data/Dashcams"
    list_files = os.listdir(dir_path)
    sum_ = 0
    std =0
    toT = torchvision.transforms.ToTensor()
    for file in list_files:
        impage_path = os.path.join(dir_path,file)
        image = cv2.imread(impage_path, cv2.IMREAD_GRAYSCALE)
        image_t = toT(image)
        # image = Image.open(impage_path)
        sum_+= torch.mean(image_t)
        std+= torch.std(image_t)
         

    print(sum_/len(list_files))
    print(std/len(list_files))