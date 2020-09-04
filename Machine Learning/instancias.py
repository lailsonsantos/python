# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:32:26 2020

@author: lailson
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from scipy import stats

iris = datasets.load_iris()
stats.describe(iris.data)

previsores = iris.data
classe = iris.target

x_treino, x_teste, y_treino, y_teste = train_test_split(previsores,
                                                       classe,
                                                       test_size = 0.3,
                                                       random_state = 0)

knn = KNeighborsRegressor(n_neighbors = 3)
knn.fit(x_treino, y_treino)

previsoes = knn.predict(x_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)
