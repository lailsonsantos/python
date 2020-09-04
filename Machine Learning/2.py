# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:54:17 2020

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

# Fazendo o carregamento dos dados diretamente do UCI Machine Learning
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

url="http://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"

# Definindo o nome de cada coluna dos dados
#names = ['age','sex','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A3','A3','A3','A3','A3','A3','A3','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','A2','A3','class']
names = ['age','sex','height	weight','QRSduration','PRinterval','Q-Tinterval','Tinterval','Pinterval','QRS','T','P','QRST','J','heartrate','chDI_Qwave','chDI_Rwave','chDI_Swave','chDI_RPwave','chDI_SPwave','chDI_intrinsicReflecttions','chDI_RRwaveExists','chDI_DD_RRwaveExists','chDI_RPwaveExists','chDI_DD_RPwaveExists','chDI_RTwaveExists','chDI_DD_RTwaveExists','chDII_Qwave','chDII_Rwave','chDII_Swave','chDII_RPwave','chDII_SPwave','chDII_intrinsicReflecttions','chDII_RRwaveExists','chDII_DD_RRwaveExists','chDII_RPwaveExists','chDII_DD_RPwaveExists','chDII_RTwaveExists','chDII_DD_RTwaveExists','chDIII_Qwave','chDIII_Rwave','chDIII_Swave','chDIII_RPwave','chDIII_SPwave','chDIII_intrinsicReflecttions','chDIII_RRwaveExists','chDIII_DD_RRwaveExists','chDIII_RPwaveExists','chDIII_DD_RPwaveExists','chDIII_RTwaveExists','chDIII_DD_RTwaveExists','chAVR_Qwave','chAVR_Rwave','chAVR_Swave','chAVR_RPwave','chAVR_SPwave','chAVR_intrinsicReflecttions','chAVR_RRwaveExists','chAVR_DD_RRwaveExists','chAVR_RPwaveExists','chAVR_DD_RPwaveExists','chAVR_RTwaveExists','chAVR_DD_RTwaveExists','chAVL_Qwave','chAVL_Rwave','chAVL_Swave','chAVL_RPwave','chAVL_SPwave','chAVL_intrinsicReflecttions','chAVL_RRwaveExists','chAVL_DD_RRwaveExists','chAVL_RPwaveExists','chAVL_DD_RPwaveExists','chAVL_RTwaveExists','chAVL_DD_RTwaveExists','chAVF_Qwave','chAVF_Rwave','chAVF_Swave','chAVF_RPwave','chAVF_SPwave','chAVF_intrinsicReflecttions','chAVF_RRwaveExists','chAVF_DD_RRwaveExists','chAVF_RPwaveExists','chAVF_DD_RPwaveExists','chAVF_RTwaveExists','chAVF_DD_RTwaveExists','chV1_Qwave','chV1_Rwave','chV1_Swave','chV1_RPwave','chV1_SPwave','chV1_intrinsicReflecttions','chV1_RRwaveExists','chV1_DD_RRwaveExists','chV1_RPwaveExists','chV1_DD_RPwaveExists','chV1_RTwaveExists','chV1_DD_RTwaveExists','chV2_Qwave','chV2_Rwave','chV2_Swave','chV2_RPwave','chV2_SPwave','chV2_intrinsicReflecttions','chV2_RRwaveExists','chV2_DD_RRwaveExists','chV2_RPwaveExists','chV2_DD_RPwaveExists','chV2_RTwaveExists','chV2_DD_RTwaveExists','chV3_Qwave','chV3_Rwave','chV3_Swave','chV3_RPwave','chV3_SPwave','chV3_intrinsicReflecttions','chV3_RRwaveExists','chV3_DD_RRwaveExists','chV3_RPwaveExists','chV3_DD_RPwaveExists','chV3_RTwaveExists','chV3_DD_RTwaveExists','chV4_Qwave','chV4_Rwave','chV4_Swave','chV4_RPwave','chV4_SPwave','chV4_intrinsicReflecttions','chV4_RRwaveExists','chV4_DD_RRwaveExists','chV4_RPwaveExists','chV4_DD_RPwaveExists','chV4_RTwaveExists','chV4_DD_RTwaveExists','chV5_Qwave','chV5_Rwave','chV5_Swave','chV5_RPwave','chV5_SPwave','chV5_intrinsicReflecttions','chV5_RRwaveExists','chV5_DD_RRwaveExists','chV5_RPwaveExists','chV5_DD_RPwaveExists','chV5_RTwaveExists','chV5_DD_RTwaveExists','chV6_Qwave','chV6_Rwave','chV6_Swave','chV6_RPwave','chV6_SPwave','chV6_intrinsicReflecttions','chV6_RRwaveExists','chV6_DD_RRwaveExists','chV6_RPwaveExists','chV6_DD_RPwaveExists','chV6_RTwaveExists','chV6_DD_RTwaveExists','chDI_JJwaveAmp','chDI_QwaveAmp','chDI_RwaveAmp','chDI_SwaveAmp','chDI_RPwaveAmp','chDI_SPwaveAmp','chDI_PwaveAmp','chDI_TwaveAmp','chDI_QRSA','chDI_QRSTA','chDII_JJwaveAmp','chDII_QwaveAmp','chDII_RwaveAmp','chDII_SwaveAmp','chDII_RPwaveAmp','chDII_SPwaveAmp','chDII_PwaveAmp','chDII_TwaveAmp','chDII_QRSA','chDII_QRSTA','chDIII_JJwaveAmp','chDIII_QwaveAmp','chDIII_RwaveAmp','chDIII_SwaveAmp','chDIII_RPwaveAmp','chDIII_SPwaveAmp','chDIII_PwaveAmp','chDIII_TwaveAmp','chDIII_QRSA','chDIII_QRSTA','chAVR_JJwaveAmp','chAVR_QwaveAmp','chAVR_RwaveAmp','chAVR_SwaveAmp','chAVR_RPwaveAmp','chAVR_SPwaveAmp','chAVR_PwaveAmp','chAVR_TwaveAmp','chAVR_QRSA','chAVR_QRSTA','chAVL_JJwaveAmp','chAVL_QwaveAmp','chAVL_RwaveAmp','chAVL_SwaveAmp','chAVL_RPwaveAmp','chAVL_SPwaveAmp','chAVL_PwaveAmp','chAVL_TwaveAmp','chAVL_QRSA','chAVL_QRSTA','chAVF_JJwaveAmp','chAVF_QwaveAmp','chAVF_RwaveAmp','chAVF_SwaveAmp','chAVF_RPwaveAmp','chAVF_SPwaveAmp','chAVF_PwaveAmp','chAVF_TwaveAmp','chAVF_QRSA','chAVF_QRSTA','chV1_JJwaveAmp','chV1_QwaveAmp','chV1_RwaveAmp','chV1_SwaveAmp','chV1_RPwaveAmp','chV1_SPwaveAmp','chV1_PwaveAmp','chV1_TwaveAmp','chV1_QRSA','chV1_QRSTA','chV2_JJwaveAmp','chV2_QwaveAmp','chV2_RwaveAmp','chV2_SwaveAmp','chV2_RPwaveAmp','chV2_SPwaveAmp','chV2_PwaveAmp','chV2_TwaveAmp','chV2_QRSA','chV2_QRSTA','chV3_JJwaveAmp','chV3_QwaveAmp','chV3_RwaveAmp','chV3_SwaveAmp','chV3_RPwaveAmp','chV3_SPwaveAmp','chV3_PwaveAmp','chV3_TwaveAmp','chV3_QRSA','chV3_QRSTA','chV4_JJwaveAmp','chV4_QwaveAmp','chV4_RwaveAmp','chV4_SwaveAmp','chV4_RPwaveAmp','chV4_SPwaveAmp','chV4_PwaveAmp','chV4_TwaveAmp','chV4_QRSA','chV4_QRSTA','chV5_JJwaveAmp','chV5_QwaveAmp','chV5_RwaveAmp','chV5_SwaveAmp','chV5_RPwaveAmp','chV5_SPwaveAmp','chV5_PwaveAmp','chV5_TwaveAmp','chV5_QRSA','chV5_QRSTA','chV6_JJwaveAmp','chV6_QwaveAmp','chV6_RwaveAmp','chV6_SwaveAmp','chV6_RPwaveAmp','chV6_SPwaveAmp','chV6_PwaveAmp','chV6_TwaveAmp','chV6_QRSA','chV6_QRSTA','class']

dataset = pandas.read_csv(url, names=names)


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
dataset.plot(kind='box', subplots=True, layout=(17,17), sharex=False, sharey=False)
plt.show()

print("Criando histogramas dos dados por classes")
dataset.hist()
plt.show()

print("Criando graficos de dispersao dos dados")
scatter_matrix(dataset)
plt.show()