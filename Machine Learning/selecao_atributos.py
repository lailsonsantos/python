# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:31:47 2020

@author: lailson
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

credito = pd.read_csv('Credit.csv')
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
                                                        test_size=(0.3), 
                                                        random_state = 0)

svm = SVC()
svm.fit(x_treino, y_treino)
previsores = svm.predict(x_teste)
taxa_acerto = accuracy_score(y_teste, previsores)

forest = ExtraTreesClassifier()
forest.fit(x_treino, y_treino)
importancia = forest.feature_importances_

x_treino2 = x_treino[:,1:12]
x_teste2 = x_teste[:,1:12]

svm2 = SVC()
svm2.fit(x_treino2, y_treino)
previsores2 = svm2.predict(x_teste2)
taxa_acerto2 = accuracy_score(y_teste, previsores2)
