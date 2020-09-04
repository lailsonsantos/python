# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:20:22 2020

@author: lailson

Não precisa informar a quantidade de clusters
"""


from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

# Carrega o dataframe Iris da biblioteca sklearn
iris = datasets.load_iris()

# Carrega os dados do dataframe Iris
cluster = kmedoids(iris.data[:, 0:2], [3, 12, 20])
cluster.get_medoids()

# Cria os clusters
cluster.process()

# Mostra os clusters criados
previsoes = cluster.get_clusters()

# Mostra as medoids criadas - pontos centrais
medoides = cluster.get_medoids()

# Cria variável para visualizar os cluster
visual = cluster_visualizer()
visual.append_clusters(previsoes, iris.data[:, 0:2])
visual.append_cluster(medoides, iris.data[:, 0:2], marker = '*', markersize = 15)
visual.show()

# Criar lista com as previsoes
lista_previsoes = []
lista_real = []

# Popular a lista com o valores das previsoes
for i in range(len(previsoes)):
    print("------")
    print(i)
    print("------")
    for j in range(len(previsoes[i])):
        print(previsoes[i][j])
        lista_previsoes.append(i)
        lista_real.append(iris.target[previsoes[i][j]])
    
# Transforma as lista em forma de array
lista_previsoes = np.asarray(lista_previsoes)    
lista_real = np.asarray(lista_real)    
    
# Matriz de confusão    
confusao = confusion_matrix(lista_previsoes, lista_real)