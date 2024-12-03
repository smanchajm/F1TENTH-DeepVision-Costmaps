# Script final presentation

### Forme de la vidéo 
- entre présentation slide et vidéo de présentation
- montrer à quoi ressemble le simulator autodrive et des F1 Tenth 

### **1. Introduction (Mise en situation)**  

- Capture l’attention avec une anecdote ou une question engageante
    - exemple : parler A2RL (Abu Dhabi autonomous racing league) -> première course à haute vitesse de voiture autonomes de courses (type f1) -> problème budget pour ce genre de voitures est énorme avec l'ensemble de l'équipement (plusieurs millions)
    - Comment rendre la conduite autonome agressive accessible ? (utilisation d'une dashcam)

- Présentez brièvement le projet :  
  - Vision-based autonomous racing.  
  - Défi principal : remplacer LiDAR par des caméras pour rendre les solutions plus accessibles et efficaces.  

---

### **2. Background (Contexte et travaux existants)**  
- Introduction à F1Tenth :  
  - Présentation de la compétition 
  - Focus sur l’utilisation traditionnelle de LiDAR dans les F1 Tenth 
- Résumé des approches précédentes :  
  - Drews et al. (2017) : MPC basé sur CNN.  
  - Cai et al. (2021) : Reinforcement Learning, limité pour la haute vitesse.  
- Limites actuelles : coût élevé, complexité, faible vitesse avec solutions vision-based.  

---

### **3. Methodology (Méthodologie)**  
  - **Collecte de données** : AutoDRIVE Simulator pour créer des datasets (images caméra → cost map).  
  - **Modèle CNN** : Entraînement pour convertir les images en cost maps exploitables.  
    - Architecture : 9 couches de convolution (afficher figure arch) + 1 couche de sortie
    - input dashcam -> output costmap birdview 
  - **MPPI** : Utilisation de MPC pour évaluer plusieurs trajectoires, choisir la meilleure et traduire cette trajectoire en consignes précises pour la voiture (throtle, orientation...)  


---

### **4. Results (Résultats)**  
- **Performance de notre modèle** :  
  - stat notre cnn
  - génération costmap
- **MPPI**

--> Optimisation toujours en cours au moment de la vidéo, des améliorations entre la vidéo et la présentation ! 

---

### **5. Conclusion**  
- Résumé des contributions :  
  - Vision-based racing viable à haute vitesse.  
  - Pipeline combinant deep learning (CNN) et MPC.  
  - Réduction des coûts grâce à l’utilisation de caméras.  
- Perspectives :  
  - Améliorations futures (robustesse, tests réels).  

- Remerciements et transition vers la session de questions.  
