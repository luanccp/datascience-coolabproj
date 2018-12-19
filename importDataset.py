
import pandas as pd
import numpy as np
#Fazer os graficos
import matplotlib.pyplot as plt
#para dividir os dados entre teste e treino
from sklearn.model_selection import train_test_split
#para regressao
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.datasets import make_regression


file = pd.ExcelFile('test.xlsx')
df = pd.read_excel('test.xlsx', sheet_name="Planilha1")

print('DADOS\n',  df)

# Metodo de normalizar na mão
df_normalizado = (df - df.mean())/ (df.max() - df.min())
print('DADOS NORMALIZADOS 1\n', df_normalizado)
X, Y = make_regression(df_normalizado)
plt.scatter(X,Y)
plt.show()


# Metodo do sklearn para normalizar
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalizado2 = pd.DataFrame(np_scaled)
#print('DADOS NORMALIZADOS 2\n', df_normalizado2)













# DESCREVE AS CARACTERISTICAS DA TABELA -> df_normalizado.describe()