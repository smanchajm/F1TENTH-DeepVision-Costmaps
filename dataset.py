import numpy as np 
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

############### versions utilisées ###########

# numpy = 1.18.5
# scipy = 1.4.1
# matplotlib = 3.2.3
# pillow = 7.2.0

###############################################

# version pour image en couleur




def rotateImage(img, angle, pivot):
    # lien de ce que j'ai copié : https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX,[0, 0]], 'constant',constant_values=img[0,0,0])
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1],:]

def cropImage(img,L,l,point):
    xp,yp = point
    imgC = img[yp-l//2:yp+l//2,xp:xp+L] # je prends le rectangle  à droite du point avec yp le centre du bord gauche du rectangle
    return imgC

def point_angle_to_map(point,angle,img):

    largeur = 100
    longueur = 100
    imgR = rotateImage(img,np.rad2deg(angle),point) # rotate dans l'angle de la direction avant de crop de manière "droite" avec le rectangle droit
    imgC = cropImage(imgR,largeur,longueur,point) # crop le rectangle droit
    imgR2 = ndimage.rotate(imgC,90,reshape = False) # pour la remettre dans la direction de la voiture
    return imgR2

def coordonnes_to_point(xabs,yabs):
    # paramètres pris sur autodrive a la mano
    ZEROX = 681
    ZEROY = 285
    UNITEX = -85
    UNITEY = -85
    return int(ZEROX +xabs * UNITEX),int(ZEROY +yabs* UNITEY)

def cardata_to_mapsurvey(xabs,yabs,angle,img):
    angle -= np.pi/2 # par rapport a auto_drive
    xp,yp = coordonnes_to_point(xabs,yabs)
    survey_image_croped = point_angle_to_map((xp,yp),angle,img)
    return survey_image_croped


################ test ###################

def main():

    image = Image.open('surveyed_map.jpg')
    img = np.array(image)

    imgshow = cardata_to_mapsurvey(-0.19,0.17,6.25,img)
    plt.figure()
    plt.imshow(imgshow)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()




########### Brouillon ###############

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

