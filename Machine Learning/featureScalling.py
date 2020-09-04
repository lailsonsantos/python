# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:15:21 2020

@author: lailson
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dataset = pd.read_csv("Dados/Credit.csv")
dt = dataset.iloc[:,[1, 4, 7]].values

# Padronização dos dados centralizando em 0 (zero), e usa o desvio padrão
standardScaler = StandardScaler()
x = standardScaler.fit_transform(dt)

# Normaliza os dados entre 0 (zero) e 1 (um)
minMax = MinMaxScaler()
y = minMax.fit_transform(dt)