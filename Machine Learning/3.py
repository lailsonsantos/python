# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:54:45 2020

@author: lailson
"""


#Disciplina: Solucoes de Mineracao de dados
#--------------------------------------------------------
#Script para a analise exploratoria dos dados (AED)
#--------------------------------------------------------


# Importando as bibliotecas necessarias
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]

# Especificando a URL da base de dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

#definindo os nomes de cada coluna
names = [ 'Id number', 'efractive index', 'Magnesium', 'Aluminum', 'Silicon', 
          'Potassium', 'Calcium', 'Barium', 'Iron', 'class']

#Fazendo o carregamento dos dados diretamente do UCI Machine Learning          
dataset = pandas.read_csv(url, names=names)

#remove a primeira coluna que contem o indice
dataset.drop([dataset.columns[0]], axis=1, inplace=True)

print("Apresentando o shape dos dados (dimenssoes)")
print(dataset.shape)

print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente"
		"os 20 primeiros registros (head(20))")
print(dataset.head(20))

print("Conhecendo os dados estatisticos dos dados carregados (describe)")
print(dataset.describe())

print("Conhecendo a distribuicao dos dados por classes (class distribution)")
print(dataset.groupby('class').size())

print("Criando grafios de caixa da distribuicao das classes")
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

print("Criando histogramas dos dados por classes")
dataset.hist()
plt.show()

print("Criando graficos de dispersao dos dados com paleta de cores")
colors_palette = {1: 'red', 2: 'yellow', 3: 'green', 5: 'black', 6: 'blue', 7: 'gray'}
colors = [colors_palette[c] for c in dataset['class']]
scatter_matrix(dataset, c=colors)
plt.show()