# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:30:27 2020

@author: lailson

Informa que o item analisado pertence a apenas 1 grupo em específico
"""

# Importa as bases de dados já prontas do sklearn
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Carrega o dataset iris da biblioteca sklearn
iris = datasets.load_iris()

# Cria duas variáveis para receber o tipo da iris e quantidade total de cada
unicos, quantidade = np.unique(iris.target, return_counts='T')

# Cria os clusters para agrupar os dados com o algoritmo Kmeans
cluster = KMeans(n_clusters = 3)
cluster.fit(iris.data)

# Cria o ponto central (médio)
centroides = cluster.cluster_centers_

# Faz a previsão da classe (target)
previsao = cluster.labels_

# Cria as variáveis previsoras
unicos2, quantidade2 = np.unique(previsao, return_counts='T')

# Matriz de confusão
matriz = confusion_matrix(iris.target, previsao)

# Plotar gráfico de pontos do dataFrame iris.data
# verificando em cada linha a condição 'previsao == 0'
# 0 - coluna do X, 1 - coluna do Y no gráfico
plt.scatter(iris.data[previsao == 0, 0], iris.data[previsao == 0, 1], 
            c = "green" , label = "setosa")
plt.scatter(iris.data[previsao == 1, 0], iris.data[previsao == 1, 1], 
            c = "red" , label = "versicolor")
plt.scatter(iris.data[previsao == 2, 0], iris.data[previsao == 2, 1], 
            c = "blue" , label = "virginica")

# Cria uma legenda
plt.legend()
