Determination de l'angle de la caméra à partir de la hauteur et d'une coordonnée : 
  Il faut au préalable obtenir les coordonnées du point en bas à gauche ainsi que du point central au fond de la table : 
X0,Y0 les coordonnées du premier et Xtab,Ytab les coordonnées du second. 

On initialise la valeur à -90° et on calcul Ytab avec cette valeur. 
On rentre ensuite dans une boucle while : On y change l'angle jusqu'à ce que les deux valeurs de Ytab soient les mêmes. 

La fonction retourne la valeur de l'angle à 0,1° près mais il est tout à fait possible d'augmenter la précision. 
