############################
### Fonctions Graphiques ###
############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm, gamma

from Variables import *

" En voyant le nombre de fois où nous allons repeter les mêmes  "
" lignes de code, on va faire des fonctions principales pour    "
" l'écriture de graphiques afin de mieux ranger notre code      "

    
def graphic_plt_id (dictionnaire, nPer, nVar, pc, PC, titre, titre_variable, xlabel, ylabel, SCT):
    DEUX=False
    nFig, axs = plt.subplots(nPer, nVar, figsize=(20,14))
    axs = np.atleast_2d(axs)
    
    nFig.suptitle(titre, fontsize=16)
    titres = list(dictionnaire.keys())
    matrices = list(dictionnaire.values())

    for i, mat in enumerate(matrices):
        row = (i) % nPer
        col = int(i/nPer)

        vect0 = mat[:,0]
        vect1 = mat[:,1]
        
        if mat.shape[1] > 2:
            vect2 = mat[:,2]
            vect3 = mat[:,3]
        # Cela voudrait dire qu'on veut plus d'une courbe dans chaque graphique
        axs[row,col].plot(vect0, vect0, label="id", color='blue')
        if SCT == True:
            axs[row,col].scatter(vect0, vect1, label=titre_variable, color='red')
            if mat.shape[1] > 2:
                axs[row,col].scatter(vect2, vect3, label="avant", color='gray')
        else:
            axs[row,col].plot(vect0, vect1, label=titre_variable, color='red')
            if mat.shape[1] > 2:
                axs[row,col].plot(vect2, vect3, label="avant", color='gray')
        if PC == True:
            axs[row,col].plot(vect0, np.full((len(mat),), pc[i]), label="pC", color='green')
        axs[row,col].set_title(titres[i])
        axs[row,col].set_xlabel(xlabel)
        axs[row,col].set_ylabel(ylabel)
        axs[row,col].legend() 
        axs[row,col].grid(True)

    plt.tight_layout()
    plt.show()


def graphic_bar (dictionnaire, nPer, nVar, titre, titre_variable, xlabel, ylabel):
    nFig, axs = plt.subplots(nPer, nVar, figsize=(20,14))
    axs = np.atleast_2d(axs)

    nFig.suptitle(titre, fontsize=16)
    titres = list(dictionnaire.keys())
    matrices = list(dictionnaire.values())

    for i, (mat) in enumerate(matrices):
        row = (i) % nPer
        col = int(i/nPer)

        vect1 = mat[:,0]
        vect2 = mat[:,1]

        axs[row,col].bar(vect1, vect2, label=titre_variable)
        axs[row,col].set_title(titres[i])
        axs[row,col].set_xlabel(xlabel)
        axs[row,col].set_ylabel(ylabel)
        axs[row,col].legend() 
        axs[row,col].grid(True)

    plt.tight_layout()
    plt.show()

# Fonction pour tracer des distributions

def graphic_pdf_cdf (dictionnaire, nPer, nVar, titre, titre_variable, xlabel, ylabel, CDF):

    nFig, axs = plt.subplots(nPer, nVar)
    axs = np.atleast_2d(axs)

    nFig.suptitle(titre, fontsize=16)
    titres = list(dictionnaire.keys())
    series = list(dictionnaire.values())

    for i, ser in enumerate(series):
        if nPer == 1:
            row = 0
            col = i
        else:
            row = (i) % nPer
            col = int(i/nPer)
    
        if CDF==True:
            n = len(ser)
            valeurs_cdf = np.sort(ser)
            cdf = np.arange(1, n+1) / n
            axs[row, col].plot(valeurs_cdf, cdf, label=titre_variable, color='red')

        else:
            axs[row,col].hist(ser,
                bins=50,
                density=True,
                alpha=0.7,
                color='red',
                label=titre_variable
            )
        axs[row,col].set_title(titres[i])
        axs[row,col].set_xlabel(xlabel)
        axs[row,col].set_ylabel(ylabel)
        axs[row,col].legend() 
        axs[row,col].grid(True)

    plt.tight_layout()
    plt.show()
