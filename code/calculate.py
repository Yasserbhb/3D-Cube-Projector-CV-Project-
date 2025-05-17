import numpy as np



def estimerH(ptImg, ptWorld):
    A = []
    for i in range(len(ptImg)):
        xw, yw = ptWorld[i]
        xi, yi = ptImg[i]
        A.append([0, 0, 0, -xw, -yw, -1, xi * xw, xi * yw, xi * 1])
        A.append([xw, yw, 1, 0, 0, 0, -yi * xw, -yi * yw, -yi])

    U, S, V = np.linalg.svd(A)
    H = np.array(V[-1].reshape((3, 3)))

    return H / H[2, 2]  # Normalize H



def calcule_P(K,H):

    RT = np.matmul(np.linalg.inv(K), H).transpose()
    #Extraire les deux premières lignes de la matrice RT comme matrice de rotation
    r1 = RT[0]
    r2 = RT[1]
    #La troisième matrice de rotation est obtenue par le produit vectoriel des deux premières matrices de rotation
    r3 = np.cross(r1, r2)
    #Assurer la droiture de la matrice de rotation
    if np.linalg.det(RT) > 0:
        RT *= -1
    #Calculer l'alpha
    alpha = np.linalg.det([r1, r2, r3])
    alpha = alpha ** (1 / float(4))
    #Normalisez pour vous assurer que la longueur du vecteur de rotation est de 1.
    r1 = r1 / alpha
    r2 = r2 / alpha
    #Utilisez le carré, car r3 est le résultat du produit croisé de r1 et r2
    r3 = r3 / (alpha * alpha)
    t = RT[2] / alpha
    P = np.array([r1, r2, r3, t]).transpose()
    # print(RT)

    return alpha,P