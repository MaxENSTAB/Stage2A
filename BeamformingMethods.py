import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd      #Imports d'usage
from scipy.signal import find_peaks
import time as t
import BeamformingNotebook
####################################################################################################
#  ATTENTION, il faut revoir ce que je mets en entrée, et surtout penser à la portée des var
####################################################################################################


# Bref ça marche pas encore 


def CBF(s_angles,s_amp,sig_n,N,M,S,L,lam):
	"""
	Classical beamforming, renvoie PCBF
	"""

	temp = rd.randn(S,L)
	n = sig_n*(rd.randn(M,L))  #Gaussien centré
	s = np.sin(2*np.pi*c0*temp/4) #Je génère un signal aléatoire
	for i in range(len(s_amp)):
	    s[i] *= s_amp[i]
	A = compute_A(s_angles,lam,M)
	y = A@s + n     #On retrouve bien la linéarité

	angles1 = np.linspace(- np.pi/2,np.pi/2,N)
	PCBF = np.zeros(N, dtype = complex)   #Ça signifie Puissance pour du classic beamforming 
	                                        # Un pic de puissance indiquera la position de notre source

	for i in range(N):
	    a_pcbf = a(angles1[i], lam,M)
	    PCBF[i] = a_pcbf.T.conj() @ Syy @ a_pcbf / (np.linalg.norm(a_pcbf) ** 4)

	return PCBF

