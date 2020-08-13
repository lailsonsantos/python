# %% [markdown]
# # Analise exploratória do Dataframe Bank (http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) 

# Importar bibliotecas
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import tree, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import Normalizer, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

plt.rcParams['figure.figsize'] = [16, 10]
pd.set_option('display.max_columns', None)

# Importar arquivo CSV com Pandas
df_csv = pd.read_csv("dados/bank.csv", sep = ";")

# Primeiros 5 registros do dataframe
df_csv.head()

# Quantidade de linhas e colunas do Dataframe - np é da bilbioteca numpy
np.shape(df_csv)