"""
Author: Maxime BARRET
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd

def a(theta,lam,M):
    """
    Fonction renvoyant le steering vector
    :param theta: angle thêta
    :param lam: paramètre de Nyquist
    :param M: nombre de capteurs (Donc ici nombre d'émetteurs)
    :return: array
    """
    res = np.zeros(M, dtype=complex)
    for i in range(0, M):
        res[i] = np.exp(-1j * (2 * np.pi * d * i * np.sin(theta) / lam))
    return res

def DP(theta):
    """
    Schéma directionnel  (directivity pattern)
    :param theta:
    :return:
    """
    A = a(theta,lam,M)
    temp = np.dot(np.conj(A).T, a(theta0,lam,M))
    dp = (temp/(np.linalg.norm(a(theta,lam,M))**2))
    return dp

if __name__ == '__main__':

    d= 0.5               #Distance entre mes émetteurs
    M = 2               #Nombre d'émetteurs sur ma cible
    N = 721
    angle = np.linspace(-np.pi/2,np.pi/2,N)     #Grille sur laquel on va chercher l'angle d'incidence

    theta0 = np.pi/4    #Angle auquel se situe le radar qu'on doit tromper
    lam = 1

    # plt.figure()
    # ax = plt.subplot(111,projection='polar')
    # DP01 = []
    # for i in range(N):
    #     DP01.append(DP(angle[i]))
    # plt.title(f"lambda = {lam} , theta0 = {theta0}")
    # plt.polar(angle, DP01)
    # plt.show()

