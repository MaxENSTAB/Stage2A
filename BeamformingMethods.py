import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd      #Imports d'usage
import scipy
from scipy.signal import find_peaks

import time as t
####################################################################################################
#  ATTENTION, il faut revoir ce que je mets en entrée, et surtout penser à la portée des var
####################################################################################################



def CBF(s_angles,s_amp,sig_n,N,M,S,L,lam,V):
	"""
	Classical beamforming, renvoie PCBF
	"""
	s = []
	for i in range(S):
	   temp = scipy.signal.square(4*np.pi*c0*np.linspace(i,L+i,L)/lam)
	   s.append(temp)

	s = np.array(s)
	
	n = sig_n*(rd.randn(M,L))  #Gaussien centré
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

	return np.array(PCBF)/np.max(PCBF)


def MUSIC(s_angles,s_amp,sig_n,N,M,S,L,lam,V):
	"""
	MUSIC

	"""
	
	s = []
	for i in range(S):
	   temp = scipy.signal.square(4*np.pi*c0*np.linspace(i,L+i,L)/lam)
	   s.append(temp)

	s = np.array(s)
	n = sig_n*(rd.randn(M,L))  #Gaussien centré
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
	s = []
	for i in range(S):
	   temp = scipy.signal.square(4*np.pi*c0*np.linspace(i,L+i,L)/lam)
	   s.append(temp)

	s = np.array(s)
	n = sig_n*(rd.randn(M,L))  #Gaussien centré
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

def DP(theta,lam,M,theta0):
	"""
	Schéma directionnel  (directivity pattern)
	:param theta:
	:return:
	"""
	A = a(theta,lam,M)
	temp = np.dot(np.conj(A).T, a(theta0,lam,M))
	dp = 20*np.log10(temp/(np.linalg.norm(a(theta,lam,M))**2))
	return dp

def C_matrix(nbCibles,AmpEcho,init_pos,v0,T):
    """
    Cette fonction va génerer deux matrices : une dont les lignes sont les c_angles précédents, l'autre dont les 
    lignes sont les c_amp précédents.
    init_pos est sous la forme : 
    [[x1,..,xn],[y1,...,yn]] --> rs = (xs-x)**2  + (ys-y)**2 et thetas = arctan((x1-v0*t)/(y1-y0))
    v0 : vitesse
    T : nombres de fois où on fait appel aux fonctions CBF, MUSIC et MVDR
    :param nbCibles: int
    :param AmpEcho: (nbCibles,)-array
    :param init_pos: 2D-array
    :param v0: float
    :param T: int
    :return: 2 *  2D-arrays
    """
    
    if len(init_pos[0]) != nbCibles:
        warnings.warn(f' {len(init_pos[0])} is different from {nbCibles}')
    
    C_ang_mat = np.zeros((T,nbCibles))
    C_amp_mat = np.zeros((T,nbCibles))
    xs = init_pos[0]
    ys = init_pos[1]
    
    for c in range(nbCibles):
        x,y = init_pos[0][c],init_pos[1][c]
        C_ang_mat[0][c] = np.arctan(x/y)
        C_amp_mat[0][c] = AmpEcho[c] *(x**2 + y**2)**(-1/2)
        
    for t in range(1,T):
        for c in range(nbCibles):
            C_ang_mat[t][0] = np.arctan((xs[c]-v0*t)/ys[c])
            C_amp_mat[t][0] = AmpEcho[c]* ((xs[c]-v0*t)**2 + ys[c]**2)**(1/2)
            C_ang_mat[t][1] = np.arctan((xs[c]+v0*t)/ys[c])
            C_amp_mat[t][1] = AmpEcho[c]* ((xs[c]+v0*t)**2 + ys[c]**2)**(1/2)
    
    return np.array(C_amp_mat), np.array(C_ang_mat)



def HMFW(plot):
    """
    A partir d'une courbe de PCBF, renvoie la liste des couples d'abscisses délimitant les extrémités d'un pic.
    :param plot: 1D array of CBF values
    :return: list of abscissa where the plot is equal to 0.5 
    """
    temp = []
    for i,value in enumerate(np.abs(plot)):
        if np.abs(value-0.5)<5e-3 and angle[i-1] not in temp:
            temp.append(angle[i])
    l = len(temp)
    print(l)
    return temp


def transmit(directions,M1,L):
    """
    Afin de coller au sujet, mais d'être plus proche de la réalité, il va falloir que je change de moyen de simulation. Ici, je ne peux pas simuler simplement l'émission de l'onde. 
    """
    pass



c0 = 3e8   #célérité de la lumière dans l'air 
d = 0.4          #Distance entre mes émetteurs
M = 5              #Nombre d'émetteurs sur ma cible
N = 4*721
angle = np.linspace(-np.pi/2,np.pi/2,N)     #Grille sur laquel on va chercher l'angle d'incidence
M1 = 10  #Nombre d'émetteurs pour le notebook BeamEmission
theta0 = 0   #Angle d'incidence auquel se situe le radar qu'on doit tromper : là, il est en face de la cible.
lam =1