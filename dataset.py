import numpy as np 
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

############### versions utilisées ###########

# numpy = 1.18.5
# scipy = 1.4.1
# matplotlib = 3.2.3
# pillow = 7.2.0

###############################################

# version pour image en couleur



track_image = Image.open('Data/TrackMap/big_surveyed_map.jpg')
track_img = np.array(track_image)
dashcam_csv = 'Data/V1 Log.csv'
dashcam_dir = 'Data/V1 Camera Frames'
costmap_dir = 'Data/Costmap'
useful_col = list(range(21))  # Only 21 first columns because we don't need values in the further columns
df = pd.read_csv(dashcam_csv, usecols=useful_col, header=None, dtype=str) # str type to prevent read_csv to parse 079 in 79 (could be a source of error for a decimal value)
   
# Get the float value from the unit part and the decimal part separated by a ',' in the csv 
def to_float(unit, decim):
    float_value = float(f"{unit}.{decim}")
    return float_value

def rotateImage(img, angle, pivot):
    # lien de ce que j'ai copié : https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX,[0, 0]], 'constant',constant_values=img[0,0,0])
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1],:]

def cropImage(img,L,l,point):
    xp,yp = point
    imgC = img[yp-l:yp,xp-L//2:xp+L//2] # Prendre le rectangle au dessus du point en centrant la dimension x sur le point
    return imgC

def point_angle_to_map(point,angle,img):
    largeur = 150
    longueur = 150
    imgR = rotateImage(img,np.rad2deg(angle),point) # rotate dans l'angle de la direction avant de crop de manière "droite" avec le rectangle droit
    imgC = cropImage(imgR,largeur,longueur,point) # crop le rectangle droit
    return imgC

def coordonnes_to_point(xabs,yabs):   # xabs correspond a la colonne 2 dans Autodrive sur les IPScoords (yabs = col1)
    # paramètres pris sur autodrive a la mano
    ZEROX = 785
    ZEROY = 359.5
    UNITEX = -85
    UNITEY = -85
    return int(ZEROX +xabs * UNITEX),int(ZEROY +yabs* UNITEY)

def cardata_to_mapsurvey(xabs,yabs,angle,img): 
    xp,yp = coordonnes_to_point(xabs,yabs)
    survey_image_croped = point_angle_to_map((xp,yp),-angle,img)
    return survey_image_croped

# add contrast to an img
def contrast_adapter(img, clip_limit, grid_size):
    img_raw = np.array(img)
    hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(hsv)
    # Applying CLAHE to L-channel clipLimit is the quantitative value for contrast and gridSize the size of the grid used to apply contrast on parts of the picture 
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size,grid_size))
    cl = clahe.apply(l)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def generate_costmap(df, img, output_dir):
    os.makedirs(output_dir, exist_ok=True) # Create costmap directory
    
    with tqdm(total = len(df)) as pb:    
        for index, row in df.iterrows():
            offset = 0
            while (int(row.iloc[5+offset]) > 3):    # col 1 to 4 of the csv can have 0 but sometimes its a decimal value so these values can take up to 8 cols
                offset += 1                         # this creates an offset for the values we're interested in. But the last value before the interesting ones has always more than 1 digit when it's not 0 (luckily) 3 is the limit because the y unit value never exceeds 3 that's how we know if we have to put an offset
            
            x_abs = to_float(row.iloc[7+offset], row.iloc[8+offset])
            y_abs = to_float(row.iloc[5+offset], row.iloc[6+offset])
            angle = to_float(row.iloc[15+offset], row.iloc[16+offset])
            
            costmap = cardata_to_mapsurvey(x_abs, y_abs, angle, img)  # generate costmap
            image_path = os.path.join(output_dir, f"costmap_{row.iloc[0]}.png")
            plt.imsave(image_path, costmap)
            pb.update()
            
#add contrast on dashcam img because half of them have too low brightness due to the simulator light settings       
def enhance_dashcam(dir):
    
    dashcams = os.listdir(dir)
    
    with tqdm(total = len(dashcams)) as pb:
        for img in dashcams:
            img_path = os.path.join(dir, img)
            dashcam_img = Image.open(img_path)
            enhanced_img = contrast_adapter(dashcam_img, 12.0, 8)
            cv2.imwrite(img_path, enhanced_img)
            pb.update()
            
def create_dataset():
    generate_costmap(df, track_img, costmap_dir)
    enhance_dashcam(dashcam_dir)
    
create_dataset()        
enhance_dashcam(dashcam_dir)

########### Brouillon ###############

# imgshow = cardata_to_mapsurvey(4.353964,1.94864,5.24789,img)
# plt.figure()
# plt.imshow(imgshow)
# plt.axis('off')
# plt.show()

# angle = -90
# xp,yp = coordonnes_to_point(-0.19,0.17)
# imgR = rotateImage(img,angle,(xp,yp))
# angle_rad = 0- np.pi/2
# xp,yp = coordonnes_to_point(-0.19,0.17)
# x_end = xp + 100* np.cos(angle_rad)
# y_end = yp +100* np.sin(angle_rad)
# imgC = cropImage(imgR,100,100,(xp,yp))
# # print(imgR.shape)
# # print(imgC.shape)
# # Supposons que 'image_array' est votre tableau NumPy contenant l'image
# # Si l'image est en niveaux de gris, utilisez 'cmap' pour afficher correctement
# plt.figure()
# plt.imshow(imgC)  # Utilisez cmap='gray' pour les images en niveaux de gris
# plt.axis('off')  # Désactive les axes pour une meilleure visibilité


# plt.figure()
# plt.plot([xp, x_end], [yp, y_end], color='red', marker='o')
# plt.imshow(img)  # Utilisez cmap='gray' pour les images en niveaux de gris
# plt.axis('off')  # Désactive les axes pour une meilleure visibilité
# plt.show()

