# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:30:39 2020

@author: lailson
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Carrega o dataframe iris
iris = sns.load_dataset('iris')

# Exibe os 5 primeiros dados do dataframe
iris.head()

# Retorna a descrição dos dados
iris.describe()

# Para verificar se existem valores faltantes, é a descrição da função describe()
iris.info()

# Mostra a relação entre os dados do dataframe
# 'hue' é usado para a coluna de rotulos
sns.pairplot(iris, hue='species')

# Exclui a coluna categorica 'species'
iris_data = iris.drop(columns='species')

# Extrai os valores numericos do dataframe e coloca na variável X
# no formato de array
x = iris_data.values

# Cria 4 cluster para treinar
km = KMeans(n_clusters = 4)
km.fit(x)

# Adiciona uma coluna ao dataframe
iris_data['labels'] = km.labels_

# Exibeo grafico com o label 'labels'
sns.pairplot(iris_data, hue='labels')

# Armazenar as siluetas do clusters - Silhouette Coefficient
sil = []

# Dispersão entre as classes
ssw = []

ch = []

# Cria mais clusters, vai de 2 a 17
lista_clusters = np.arange(2, 18)

for n_c in lista_clusters:
    km = KMeans(n_clusters=n_c).fit(x)
    
    sil.append(silhouette_score(x, km.labels_))
    ssw.append(km.inertia_)
    ch.append(calinski_harabasz)

plt.plot(lista_clusters, sil)
plt.title('silhouette')
plt.show()

plt.plot(lista_clusters, ssw)
plt.title('SSW')
plt.show()

plt.plot(lista_clusters, ch)
plt.title('ch')
plt.show()

# Normalizar os dados
normalizer = MinMaxScaler()

# Treina e normaliza os dados
x_norm = normalizer.fit_transform(x)

# Armazenar as siluetas do clusters - Silhouette Coefficient
sil = []

# Dispersão entre as classes
ssw = []

ch = []

for n_c in lista_clusters:
    km = KMeans(n_clusters=n_c).fit(x_norm)
    
    sil.append(silhouette_score(x_norm, km.labels_))
    ssw.append(km.inertia_)
    ch.append(calinsk)

plt.plot(lista_clusters, sil)
plt.title('silhouette')
plt.show()

plt.plot(lista_clusters, ssw)
plt.title('SSW')
plt.show()