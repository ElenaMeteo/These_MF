############################
### Fonctions Executives ###
############################
# commmetnqire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, truncnorm, gamma
from Variables import *
from Fonctions_Tech import prob, Np, pPrime, pC, H, F
from Fonctions_Graph import graphic_bar, graphic_pdf_cdf, graphic_plt_id

" En voyant le nombre de fois où nous allons repeter les mêmes  "
" lignes de code, on va faire des fonctions principales pour    "
" le main afin de mieux ranger notre code                       "

def mainExec(obs, prev, k, N, percentile):
    # ÉLÉMENTS DE CALCUL
    ####################
    # Condition à établi
    cond = np.percentile(obs, percentile)
    # Événement théorique
    Etheo = prev > cond
    # Événement observé
    Eobs = obs > cond
    # Vecteur de probabilités par jour
    prob_vecteur = prob(Etheo, k, N)
    # Matrice de fréquences de probabilités
    Np_matrice = Np(prob_vecteur)
    Np_matrice = np.array(Np_matrice)
    # Matrices de fréquences réelles (p'(p))
    pPrime_matrice = pPrime(Eobs, prob_vecteur, Np_matrice)
    # Probabilité climatologique
    pc = pC(Eobs, N)
    # H et F
    hit_rate = H(Eobs, prob_vecteur, k)
    false_alarm = F(Eobs, prob_vecteur, k)

    # SCORES
    ########
    # brierScore = Brier(Np_matrice, pPrime_matrice, N, pc)
    # brierSkill = BSS(Np_matrice, pPrime, N, pc)
    
    return cond, Etheo, Eobs, Np_matrice, pPrime_matrice, pc, hit_rate, false_alarm  

def graphsExec(Fiab, Acuite, ROC, dict_fiab, pc, dict_acuite, dict_ROC, per, nVar):
    
    # Titres Fiabilité
    titre_fiab = "Diagramme de fiabilité: 50 et 90"
    titre_variable_fiab = "pPrime(p)"
    xlabel_fiab = "probabilités (%)"
    ylabel_fiab = "fréquence réelle (%)"
    
    # Titres Acuité
    titre_acuite = "Diagramme d'acuité: 50 et 90"
    titre_variable_acuite= ""
    xlabel_acuite = "probabilités (%)"
    ylabel_acuite = "fréquence (%)"
    
    # Titres ROC 
    titre_ROC = "Courbe ROC: 50 et 90"
    titre_variable_ROC = "H(F)"
    xlabel_ROC = "False alarm"
    ylabel_ROC = "Hit rate"

    
    if Fiab==True:
        PC=True
        SCT=False
        graphic_plt_id(dict_fiab, len(per), nVar, pc, PC, titre_fiab, titre_variable_fiab, xlabel_fiab, ylabel_fiab, SCT)
        
    if Acuite==True:
        graphic_bar(dict_acuite, len(per), nVar, titre_acuite, titre_variable_acuite, xlabel_acuite, ylabel_acuite)
        
    if ROC==True:
        PC=False
        SCT=True
        graphic_plt_id(dict_ROC, len(per), nVar, pc, PC, titre_ROC, titre_variable_ROC, xlabel_ROC, ylabel_ROC, SCT)
