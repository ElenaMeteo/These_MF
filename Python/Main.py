###########################################
### Analyse et validation données météo ###
###########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm, gamma
from pathlib import Path


from Variables import *
from Fonctions_Exec import mainExec, graphsExec
from Fonctions_Tech import pointROC, conv_vent, sigmaMean, RMSEcheck
from Fonctions_Graph import graphic_pdf_cdf, graph_simple

np.set_printoptions(threshold=np.inf)

###########
# DONNÉES #
###########

# Carpeta donde está este archivo .py
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas a los ficheros
ruta_T  = BASE_DIR / "Data" / "donnees_test_temperature.txt"
ruta_V  = BASE_DIR / "Data" / "donnees_test_vent.txt"
ruta_Va = BASE_DIR / "Data" / "donnees_test_vent_analyse.txt"

dfT = pd.read_csv(ruta_T, sep=r'\s+', header=None)
dfV = pd.read_csv(ruta_V, sep=r'\s+', header=None)
dfVa = pd.read_csv(ruta_Va, sep=r'\s+', header=None)

obsT = dfT.iloc[:, -1]
prevT = dfT.iloc[:, 0:kT].to_numpy()

obsV = dfV.iloc[:, -1]
prevV = dfV.iloc[:, 0:kV].to_numpy()

obsVa = dfVa.iloc[:, -1]
prevVa = dfVa.iloc[:, 0:kVa].to_numpy()

condT50, EtheoT50, EobsT50, NpT50, pPrimeT50, pcT50, HT50, FT50 = mainExec(obsT, prevT, kT, NT, 50)
condT90, EtheoT90, EobsT90, NpT90, pPrimeT90, pcT90, HT90, FT90 = mainExec(obsT, prevT, kT, NT, 90)

condV50, EtheoV50, EobsV50, NpV50, pPrimeV50, pcV50, HV50, FV50 = mainExec(obsV, prevV, kV, NV, 50)
condV90, EtheoV90, EobsV90, NpV90, pPrimeV90, pcV90, HV90, FV90 = mainExec(obsV, prevV, kV, NV, 90)

condVa50, EtheoVa50, EobsVa50, NpVa50, pPrimeVa50, pcVa50, HVa50, FVa50 = mainExec(obsVa, prevVa, kVa, NVa, 50)
condVa90, EtheoVa90, EobsVa90, NpVa90, pPrimeVa90, pcVa90, HVa90, FVa90 = mainExec(obsVa, prevVa, kVa, NVa, 90)

##############
# DIAGRAMMES #
##############

# Diagramme de fiabilité
########################

pc = [pcT50, pcT90, pcV50, pcV90, pcVa50, pcVa90]


dict_fiab = {"Température_50": pPrimeT50, "Température_90": pPrimeT90,
            "Vent_obs_50": pPrimeV50, "Vent_obs_90": pPrimeV90,
            "Vent_analyse_50": pPrimeVa50, "Vent_analyse_90": pPrimeVa90}

# Diagramme d'acuité
####################

dict_acuite = {"Température_50": NpT50, "Température_90": NpT90,
                "Vent_obs_50": NpV50, "Vent_obs_90": NpV90,
                "Vent_analyse_50": NpVa50, "Vent_analyse_90": NpVa50}

# Courbe de ROC
###############

FHT50 = np.vstack((FT50, HT50)).T
FHT90 = np.vstack((FT90, HT90)).T
FHV50 = np.vstack((FV50, HV50)).T
FHV90 = np.vstack((FV90, HV90)).T
FHVa50 = np.vstack((FVa50, HVa50)).T
FHVa90 = np.vstack((FVa90, HVa90)).T

dict_ROC = {"Température_50": FHT50, "Température_90": FHT90,
                "Vent_obs_50": FHV50, "Vent_obs_90": FHV90,
                "Vent_analyse_50": FHVa50, "Vent_analyse_90": FHVa90}

Fiab=True
Acuite=True
ROC=True

# graphsExec(Fiab, Acuite, ROC, dict_fiab, pc, dict_acuite, dict_ROC, [50, 90], 3)

# Catégorie de probabilité optimale.
pointROC(dict_ROC, kT)

########################
# ERREUR D'OBSERVATION #
########################

" L'idée pour la suite est de prendre en compte les erreurs  "
" d'observation au moment de faire la prévision. Nous allons "
" faire cela de deux façons différentes: en perturbant les   "
" observations et avec les 'conditional predictands', qui    "
" assurent un comportement non biaisé et propre.             "

" Nous allons tout d'abord faire cetta analyse avec les données "
" du vent. "

" La distribution d'erreur de base est établie comme une normale "
" tronquée de laquelle on va estimer les paramètres. "

# Analyse des perturbations
###########################

avant_1DV_1l = prevV[0,:].ravel()
apres_1DV_1l = conv_vent(prevV[0,:], kV, obsV[0], delta) 
"On fait ça pour controler ligne par ligne aussi (on regarde ce qu'il se passe sur la 1ere)"
avant_1DV = prevV.ravel()
apres_1DV = np.array([
    conv_vent(prevV[i, :], kV, obsV[i], delta)
    for i in range(prevV.shape[0])
]).ravel()
avant_1DVa = prevVa.ravel()
apres_1DVa = np.array([
    conv_vent(prevVa[i, :], kVa, obsVa[i], delta)
    for i in range(prevVa.shape[0])
]).ravel()

print("check funcion:", np.array_equal(avant_1DV, apres_1DV))

# Variables de contrôle
mediaV_1l_av = np.mean(avant_1DV_1l)
varV_1l_av = np.var(avant_1DV_1l)

mediaV_av = np.mean(avant_1DV)
varV_av = np.var(avant_1DV)

mediaVa_av = np.mean(avant_1DVa)
varVa_av = np.var(avant_1DVa)

mediaV_1l_ap = np.mean(apres_1DV_1l)
varV_1l_ap = np.var(apres_1DV_1l)

mediaV_ap = np.mean(apres_1DV)
varV_ap = np.var(apres_1DV)

mediaVa_ap = np.mean(apres_1DVa)
varVa_ap = np.var(apres_1DVa)

# meanV_ap_lpl = np.mean(var_samplesV)
# meanVa_ap_lpl = np.mean(var_samplesVa)

print("\nMoyenne 1l vent obs avant/après:", mediaV_1l_av, mediaV_1l_ap)
print("Moyenne vent obs avant/après:", mediaV_av, mediaV_ap)
print("Moyenne vent ana avant/après:", mediaVa_av, mediaVa_ap)

print("\nVariance 1l vent obs avant/après:", varV_1l_av, varV_1l_ap)
print("Variance vent obs avant/après:", varV_av, varV_ap)
print("Variance vent ana avant/après:", varVa_av, varVa_ap)

media_sigmaV = sigmaMean(prevV, obsV, delta)
media_sigmaVa = sigmaMean(prevVa, obsVa, delta)

print("\nLa variances des perturbations pour vent obs est:", media_sigmaV)
print("La variances des perturbations pour vent ana est:", media_sigmaVa)

dict_all = {
    "Av pert 1l vent obs": avant_1DV_1l, 
    "Ap pert 1l vent obs": apres_1DV_1l,
    "Av pert vent obs": avant_1DV, 
    "Ap pert vent obs": apres_1DV,
    "Av pert vent ana": avant_1DVa, 
    "Ap pert vent ana": apres_1DVa
}

titre = "Distribution (pdf) des prévisions avant et aprés perturbation"
titre_variable = "Distribution de probabilité"
xlabel = "Prévisions"
ylabel = "Fréquences relatives"

graphic_pdf_cdf(dict_all, 2, 3, titre, titre_variable, xlabel, ylabel, CDF=False)

titre = "Distribution (cdf) des prévisions avant et aprés perturbation"
titre_variable = "Distribution cummulée de probabilité"
xlabel = "Prévisions"
ylabel = "Fréquences cumulées"

graphic_pdf_cdf(dict_all, 2, 3, titre, titre_variable, xlabel, ylabel, CDF=True)

"Si maintenant on réalise la même analyse que avant mais avec "
"les valeurs corrigées, on va peut-être avoir des meilleurs "
"résultats."

# Analyse avec les nouvelles valeurs
####################################

prevV_ap = apres_1DV.reshape(-1, kV)
prevVa_ap = apres_1DVa.reshape(-1, kVa)

condV50_ap, EtheoV50_ap, EobsV50_ap, NpV50_ap, pPrimeV50_ap, pcV50_ap, HV50_ap, FV50_ap = mainExec(obsV, prevV_ap, kV, NV, 50)
condV90_ap, EtheoV90_ap, EobsV90_ap, NpV90_ap, pPrimeV90_ap, pcV90_ap, HV90_ap, FV90_ap = mainExec(obsV, prevV_ap, kV, NV, 90)

condVa50_ap, EtheoVa50_ap, EobsVa50_ap, NpVa50_ap, pPrimeVa50_ap, pcVa50_ap, HVa50_ap, FVa50_ap = mainExec(obsVa, prevVa_ap, kVa, NVa, 50)
condVa90_ap, EtheoVa90_ap, EobsVa90_ap, NpVa90_ap, pPrimeVa90_ap, pcVa90_ap, HVa90_ap, FVa90_ap = mainExec(obsVa, prevVa_ap, kVa, NVa, 90)

# Diagramme de fiabilité
########################
pc_ap = [pcV50_ap, pcV90_ap, pcVa50_ap, pcVa90_ap]

two_pPrimeV50 = np.hstack((pPrimeV50_ap, pPrimeV50))
two_pPrimeV90 = np.hstack((pPrimeV90_ap, pPrimeV90))
two_pPrimeVa50 = np.hstack((pPrimeVa50_ap, pPrimeVa50))
two_pPrimeVa90 = np.hstack((pPrimeVa90_ap, pPrimeVa90))

dict_fiab_ap = {"Vent_obs_50_ap": two_pPrimeV50, "Vent_obs_90_ap": two_pPrimeV90,
                "Vent_analyse_50_ap": two_pPrimeVa50, "Vent_analyse_90_ap": two_pPrimeVa90
                }

# Diagramme d'acuité
####################

dict_acuite_ap = {"Vent_obs_50_ap": NpV50_ap, "Vent_obs_90_ap": NpV90_ap,
                "Vent_analyse_50_ap": NpVa50_ap, "Vent_analyse_90_ap": NpVa50_ap}

# Courbe de ROC
###############

FHV50_ap = np.vstack((FV50_ap, HV50_ap)).T
FHV90_ap = np.vstack((FV90_ap, HV90_ap)).T
FHVa50_ap = np.vstack((FVa50_ap, HVa50_ap)).T
FHVa90_ap = np.vstack((FVa90_ap, HVa90_ap)).T

two_FHV50 = np.hstack((FHV50_ap, FHV50))
two_FHV90 = np.hstack((FHV90_ap, FHV90))
two_FHVa50 = np.hstack((FHVa50_ap, FHVa50))
two_FHVa90 = np.hstack((FHVa90_ap, FHVa90))

dict_ROC_ap = {"Vent_obs_50_ap": two_FHV50, "Vent_obs_90_ap": two_FHV90,
            "Vent_analyse_50_ap": two_FHVa50, "Vent_analyse_90_ap": two_FHVa90}

graphsExec(Fiab, Acuite, ROC, dict_fiab_ap, pc_ap, dict_acuite_ap, dict_ROC_ap, [50, 90], 2)

# RMSE VS. ÉCART-TYPE
#####################

rmseV_av = []
rmseV_ap = []
ectV_av = []
ectV_ap = []

rmseVa_av = []
rmseVa_ap = []
ectVa_av = []
ectVa_ap = []

rmseV_av.append(RMSEcheck(prevV, obsV)[0])
ectV_av.append(RMSEcheck(prevV, obsV)[1])

rmseV_ap.append(RMSEcheck(prevV_ap, obsV)[0])
ectV_ap.append(RMSEcheck(prevV_ap, obsV)[1])

rmseVa_av.append(RMSEcheck(prevVa, obsVa)[0])
ectVa_av.append(RMSEcheck(prevVa, obsVa)[1])

rmseVa_ap.append(RMSEcheck(prevVa_ap, obsVa)[0])
ectVa_ap.append(RMSEcheck(prevVa_ap, obsVa)[1])

print("Vent obs avant: RMSE, écart type", rmseV_av, ectV_av)
print("Vent obs apres: RMSE, écart type", rmseV_ap, ectV_ap)

print("Vent ana avant: RMSE, écart type", rmseVa_av, ectVa_av)
print("Vent ana apres: RMSE, écart type", rmseVa_ap, ectVa_ap)

# Calcul des sigmas correspondents
##################################

# sigma = 10

sigma_meanV10 = sigmaMean(prevV, obsV, delta10)
sigma_meanVa10 = sigmaMean(prevVa, obsVa, delta10)

print("\nLa variances des perturbations pour vent obs et delta = 10 est:", sigma_meanV10)
print("La variances des perturbations pour vent ana et delta = 10 est:", sigma_meanVa10)

# sigma = 15

sigma_meanV15 = sigmaMean(prevV, obsV, delta15)
sigma_meanVa15 = sigmaMean(prevVa, obsVa, delta15)

print("\nLa variances des perturbations pour vent obs et delta = 15 est:", sigma_meanV15)
print("La variances des perturbations pour vent ana et delta = 15 est:", sigma_meanVa15)

# sigma = 20

sigma_meanV20 = sigmaMean(prevV, obsV, delta20)
sigma_meanVa20 = sigmaMean(prevVa, obsVa, delta20)

print("\nLa variances des perturbations pour vent obs et delta = 20 est:", sigma_meanV20)
print("La variances des perturbations pour vent ana et delta = 20 est:", sigma_meanVa20)

# RMSE vs. écart-type (variation maille)
########################################

# delta = 10

apres10_1DV = np.array([
    conv_vent(prevV[i, :], kV, obsV[i], delta10)
    for i in range(prevV.shape[0])
]).ravel()

prevV10_ap = apres10_1DV.reshape(-1, kV)
rmseV_ap.append(RMSEcheck(prevV10_ap, obsV)[0])
ectV_ap.append(RMSEcheck(prevV10_ap, obsV)[1])

# delta = 15

apres15_1DV = np.array([
    conv_vent(prevV[i, :], kV, obsV[i], delta15)
    for i in range(prevV.shape[0])
]).ravel()

prevV15_ap = apres15_1DV.reshape(-1, kV)
rmseV_ap.append(RMSEcheck(prevV15_ap, obsV)[0])
ectV_ap.append(RMSEcheck(prevV15_ap, obsV)[1])

# delta = 20

apres20_1DV = np.array([
    conv_vent(prevV[i, :], kV, obsV[i], delta20)
    for i in range(prevV.shape[0])
]).ravel()

prevV20_ap = apres20_1DV.reshape(-1, kV)
rmseV_ap.append(RMSEcheck(prevV20_ap, obsV)[0])
ectV_ap.append(RMSEcheck(prevV20_ap, obsV)[1])


# Vecteur
delta_vector = [delta, delta10, delta15, delta20]

# Rapport des deux vecteurs
rmseV_ap = np.array(rmseV_ap)
ectV_ap = np.array(ectV_ap)
rapportV_ap = rmseV_ap/ectV_ap

titre_variable = "Score RMSE en fonction du maillage"
xlabel = "Maillage (km)"
ylabel = "Rapport RMSE/ect"
graph_simple(delta_vector, rapportV_ap, titre_variable, xlabel, ylabel)