import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd      #Imports d'usage
from scipy.signal import find_peaks
import time as t
####################################################################################################
#  ATTENTION, il faut revoir ce que je mets en entrée, et surtout penser à la portée des var
####################################################################################################



# Bref ça marche pas encore 
def CBF(s_angles,s_amp,sig_n,N,M,S,L,lam,V):
	"""
	Classical beamforming, renvoie PCBF
	"""
	temp = rd.randn(S,L)
	n = sig_n*(rd.randn(M,L))  #Gaussien centré
	s = np.sin((1-np.linalg.norm(V)/c)*2*np.pi*c0*temp/4) #Je génère un signal aléatoire
	for i in range(len(s_amp)):
	    s[i] *= s_amp[i]
	A = compute_A(s_angles,lam,M)
	y = A@s + n     #On retrouve bien la linéarité
	Syy = y @ y.T.conj() / L


	angles1 = np.linspace(- np.pi/2,np.pi/2,N)
	PCBF = np.zeros(N, dtype = complex)   #Ça signifie Puissance pour du classic beamforming 
	                                        # Un pic de puissance indiquera la position de notre source

	for i in range(N):
	    a_pcbf = a(angles1[i], lam,M)
	    PCBF[i] = a_pcbf.T.conj() @ Syy @ a_pcbf / (np.linalg.norm(a_pcbf) ** 4)

	return np.array(PCBF)


def MUSIC(s_angles,s_amp,sig_n,N,M,S,L,lam,V):
	"""
	MUSIC

	"""
	temp = rd.randn(S,L)
	n = sig_n*(rd.randn(M,L))  #Gaussien centré
	s = np.sin((1-np.linalg.norm(V)/c)*2*np.pi*c0*temp/4) #Je génère un signal aléatoire
	for i in range(len(s_amp)):
	    s[i] *= s_amp[i]
	A = compute_A(s_angles,lam,M)
	y = A@s + n     #On retrouve bien la linéarité

	Syy = y @ y.T.conj() / L
	eig = np.linalg.eig(Syy)    #On calcule les valeurs propres et des vecteurs propres de la matrice de covariance
	eigvals = eig[0]
	Diagonale = np.diag(eigvals)   

	U = eig[1]
	#Où commence le sous-espace contenant le bruit? à S+1 !
	Unoise = U[:,S:]
	PMUSIC= np.zeros(N, dtype = complex)
	for i in range(N):
	    a_music = a(angle[i], lam,M)
	    PMUSIC[i] = 1 / (np.conj(a_music).T @ Unoise@np.conj(Unoise.T)@a_music)
	return np.array(PMUSIC/np.max(PMUSIC))


def MVDR(s_angles,s_amp,sig_n,N,M,S,L,lam,V):
	"""
	MVDR
	"""
	temp = rd.randn(S,L)
	n = sig_n*(rd.randn(M,L))  #Gaussien centré
	s = np.sin((1-np.linalg.norm(V)/c)*2*np.pi*c0*temp/4) #Je génère un signal aléatoire
	for i in range(len(s_amp)):
	    s[i] *= s_amp[i]
	A = compute_A(s_angles,lam,M)
	y = A@s + n     #On retrouve bien la linéarité

	Syy = y @ y.T.conj() / L
	PMVDR = np.zeros(N, dtype = complex)
	for i in range(N):
	    a_pmvdr = a(angle[i], lam,M)
	    PMVDR[i] = 1/(np.conj(a_pmvdr.T) @ np.linalg.inv(Syy) @ a_pmvdr)
	return np.array(PMVDR/np.max(PMVDR))

def DP(theta):
    """
    Schéma directionnel  (directivity pattern)
    :param theta:
    :return:
    """
    A = a(theta,lam,M)
    temp = np.dot(np.conj(A).T, a(theta0,lam,M))
    dp = 20*np.log10(temp/(np.linalg.norm(a(theta,lam,M))**2))
    return dp

def a(theta,lam,M):
    """
    Fonction renvoyant le steering vector
    :param theta: angle thêta
    :param lam: longueur d'onde
    :param M: nombre de capteurs (Donc ici nombre d'émetteurs)
    :return: array
    """
    res = np.zeros(M, dtype=complex)
    for i in range(0, M):
        res[i] = np.exp(-1j * (2 * np.pi * d * i * np.sin(theta) / lam))
    return res

def compute_A(thetas,lam,M):
    """
    Fonction renvoyant la steering matrix, qui serait la concaténation des steering vectors pour chaque theta
    :param thetas: liste d'angles
    :param lam: longueur d'onde
    :param M: nombre de capteurs (Donc ici nombre d'émetteurs)
    :return: 2D array
    """
    res = np.zeros((len(thetas), M), dtype = complex)
    
    for i in range(len(thetas)):
        lis = np.zeros(M, dtype = complex)
        for k in range(0,M):
            lis[k] = np.exp(-1j * (2*np.pi*d*k*np.sin(thetas[i]) /lam))
        res[i] = lis
    return res.T


c0 = 3e8   #célérité de la lumière dans l'air 
d = 0.4          #Distance entre mes émetteurs
M = 5              #Nombre d'émetteurs sur ma cible
N = 4*721
angle = np.linspace(-np.pi/2,np.pi/2,N)     #Grille sur laquel on va chercher l'angle d'incidence

theta0 = 0   #Angle d'incidence auquel se situe le radar qu'on doit tromper : là, il est en face de la cible.
