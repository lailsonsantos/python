# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:11:11 2020

@author: lailson
"""

# modulo importados
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

# Carrega o arquivo CSV
dataset = pd.read_csv("Dados/Credit.csv")

# Colunas usadas
# ['personal_status', 'other_parties']

# Separa as colunas ['personal_status', 'other_parties']
colunas = dataset.iloc[:,8:10].values
colunas2 = dataset.iloc[:,8:10].values

# Cria objeto LabelEncoder
labelEncoding = LabelEncoder()

# Muda as colunas ['personal_status', 'other_parties'] para números
colunas[:,0] = labelEncoding.fit_transform(colunas[:,0])
colunas[:,1] = labelEncoding.fit_transform(colunas[:,1])

# Transforma as colunas em array de inteiros
coluna_int = colunas.astype(int)

# Cria objeto OnehotEncoder
# O valor [0] é a coluna usada no processo de transformação
# Permanece os valores das demais colunas
# Cria as diferentes colunas para a quantidade de valores diferentes
onehotencoder = make_column_transformer((OneHotEncoder(categories='auto', sparse=False), [0]), remainder="passthrough")

# OnehotEncoder transforma todo o DataFrame
colunas2 = onehotencoder.fit_transform(colunas2)
