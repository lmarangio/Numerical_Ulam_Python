#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings; warnings.simplefilter('ignore')
import pandas as pd
df=pd.read_csv('interventionsEtInfos.csv')


# In[2]:


variables = [##'annee', 
                   #'heure',
               #'directionVentBale', #'directionVentDijon', 'directionVentNancy',
               ##'humiditeBale', #'humiditeDijon', 'humiditeNancy',
               #'nebulositeBale',
               #'phaseLune', 
               #'distanceLune',
               #'pointRoseeBale', 'pointRoseeDijon', 'pointRoseeNancy', 
               #'precipitations1hBale', #'precipitations1hDijon', 'precipitations1hNancy',
               #'precipitations3hBale', 'precipitations3hDijon', 'precipitations3hNancy',
               #'pressionBale', 'pressionDijon', 'pressionNancy',
               #'pressionMerBale', 'pressionMerDijon', 'pressionMerNancy',
               #'pressionVar3hBale', 'pressionVar3hDijon', 'pressionVar3hNancy',
               #'rafalesSur1perBale', 'rafalesSur1perDijon', 'rafalesSur1perNancy',
               #'temperatureBale', #'temperatureDijon', 'temperatureNancy',
               #'visibiliteBale', 'visibiliteDijon', 'visibiliteNancy',
               #'vitesseVentBale', 'vitesseVentDijon', 'vitesseVentNancy',
               #'diarrhee_inc', 
               #'diarrhee_inc100', 
               #'diarrhee_inc100_low',
               #'diarrhee_inc100_up', 'diarrhee_inc_low', 'diarrhee_inc_up',
               #'grippe_inc',
               #'grippe_inc100', 
               #'grippe_inc100_low',
               #'grippe_inc100_up', 'grippe_inc_low', 'grippe_inc_up',
               #'varicelle_inc',
               #'varicelle_inc100',
               #'varicelle_inc100_low',
               #'varicelle_inc100_up', 'varicelle_inc_low', 'varicelle_inc_up',
               #'hAllanCourcelle', 'hAllanCourcelleNb', 'hAllanCourcelleStd',
               #'hDoubsBesancon', 'hDoubsBesanconNb', 'hDoubsBesanconStd',
               #'hDoubsVoujeaucourt', 'hDoubsVoujeaucourtNb', 'hDoubsVoujeaucourtStd',
               #'hLoueOrnan', 'hLoueOrnanNb', 'hLoueOrnanStd',
               #'hOgnonBonna', 'hOgnonBonnaNb', 'hOgnonBonnaStd',
               #'hOgnonMontessau', 'hOgnonMontessauNb','hOgnonMontessauStd'
               #'bisonFuteDepart', 'bisonFuteRetour', 
               #'debutFinVacances', 
               #'ferie', 
               #'jourSemaine', 
               #'jour', 
               ##'jourAnnee',
               #'mois',
               #'luneApparente', 'nuit',
               #'tempsPresentBale', 'tempsPresentDijon', 'tempsPresentNancy', 
               #'tendanceBaromBale', 'tendanceBaromDijon', 'tendanceBaromNancy', 
               #'vacances', 'veilleFerie',
             'nbInterventions'
            ]


# In[3]:


import numpy as np


# In[4]:


def get_orbite(df, variables):
    x = np.array([df[k].values.tolist() for k in variables]).T
    return x
#    for i in range(len(x)):
#     print(x[i])
 #   return (x-x.min(axis=0)) / (x.max(axis=0)-x.min(axis=0))

x = get_orbite(df, variables)


# In[5]:



print(x.shape)
#for i in range(len(x)):
#     print(x[i])



