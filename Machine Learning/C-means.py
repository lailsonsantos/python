# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:11:25 2020

@author: lailson

Informa que o item analisado pertence a 1 ou mais grupos
"""


import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np
import skfuzzy as sk

# Carrega o banco de dados Iris da biblioteca Sklearn
iris = datasets.load_iris()

# Cria c-means
r = sk.cmeans(data = iris.data.T, c = 3, m = 2, error = 0.005,
              maxiter = 1000, init = None)

# Chances do registro pertencer a um determinado grupo
previsoes_porcentagem = r[1]

# Chances do registro na linha x coluna pertencer a um determinado grupo
previsoes_porcentagem[0][0]
previsoes_porcentagem[1][0]
previsoes_porcentagem[2][0]

# argmax retorna o maior valor
# axis informar a linha (0) ou coluna (1)
previsores = previsoes_porcentagem.argmax(axis = 0)

# Matriz de confusão
resultado = confusion_matrix(iris.target, previsores)
