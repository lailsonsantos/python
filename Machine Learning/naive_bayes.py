# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:53:18 2020

@author: lailson
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix

credito = pd.read_csv('Dados/Credit.csv')
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

x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, classe, test_size = 0.3, random_state = 0)

naive_bayes = GaussianNB()
naive_bayes.fit(x_treino, y_treino)

previsoes = naive_bayes.predict(x_teste)

confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

v = ConfusionMatrix(GaussianNB())
v.fit(x_treino, y_treino)
v.score(x_teste, y_teste)
v.poof()


novo_credito = pd.read_csv('Dados/NovoCredit.csv')
novo_credito = novo_credito.iloc[:,0:20].values

novo_credito[:,0] = labelEncoder.fit_transform(novo_credito[:,0])
novo_credito[:,2] = labelEncoder.fit_transform(novo_credito[:,2])
novo_credito[:,3] = labelEncoder.fit_transform(novo_credito[:,3])
novo_credito[:,5] = labelEncoder.fit_transform(novo_credito[:,5])
novo_credito[:,6] = labelEncoder.fit_transform(novo_credito[:,6])
novo_credito[:,8] = labelEncoder.fit_transform(novo_credito[:,8])
novo_credito[:,9] = labelEncoder.fit_transform(novo_credito[:,9])
novo_credito[:,11] = labelEncoder.fit_transform(novo_credito[:,11])
novo_credito[:,13] = labelEncoder.fit_transform(novo_credito[:,13])
novo_credito[:,14] = labelEncoder.fit_transform(novo_credito[:,14])
novo_credito[:,16] = labelEncoder.fit_transform(novo_credito[:,16])
novo_credito[:,18] = labelEncoder.fit_transform(novo_credito[:,18])
novo_credito[:,19] = labelEncoder.fit_transform(novo_credito[:,19])

naive_bayes.predict(novo_credito)
