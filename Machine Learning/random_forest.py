# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:03:40 2020

@author: lailson
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

credito = pd.read_csv("Dados/Credit.csv")
previsores = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

labelEncoder = LabelEncoder()
previsores[:,0] = labelEncoder.fit_transform(previsores[:,0])
previsores[:,2] = labelEncoder.fit_transform(previsores[:,2])
previsores[:,3] = labelEncoder.fit_transform(previsores[:,3])
previsores[:,5] = labelEncoder.fit_transform(previsores[:,5])
previsores[:,6] = labelEncoder.fit_transform(previsores[:,6])
previsores[:,8] = labelEncoder.fit_transform(previsores[:,8])
previsores[:,9] = labelEncoder.fit_transform(previsores[:,9])
previsores[:,11] = labelEncoder.fit_transform(previsores[:,11])
previsores[:,13] = labelEncoder.fit_transform(previsores[:,13])
previsores[:,14] = labelEncoder.fit_transform(previsores[:,14])
previsores[:,16] = labelEncoder.fit_transform(previsores[:,16])
previsores[:,18] = labelEncoder.fit_transform(previsores[:,18])
previsores[:,19] = labelEncoder.fit_transform(previsores[:,19])

x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, 
                                                        classe,
                                                        test_size = 0.3, 
                                                        random_state = 0)

floresta = RandomForestClassifier(n_estimators=100)
floresta.fit(x_treino, y_treino)
previsoes = floresta.predict(x_teste)

confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

floresta.estimators_

floresta.estimators_[1]
