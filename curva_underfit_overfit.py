# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:19:57 2018

@author: Luan
"""

from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#para dividir os dados entre teste e treino
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dados_iris = load_iris()

# Função que retorna os valores
def retornaResultadosModeloKNN(random_state, quantidade, dados, respostas):
    X_train, X_test, y_train, y_test = train_test_split(dados, respostas, random_state = random_state)
    #vetores de armazenamento dos resultados de testes e dos treinos
    quantidade_k = range(1, quantidade + 1)
    res_teste = []
    res_treino = []
    
    #loop das classificações
    for i in quantidade_k:
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train, y_train)
        
        #colocando os resultados nos vetores
        res_treino.append(knn.score(X_train, y_train))
        res_teste.append(knn.score(X_test, y_test))
    
    return quantidade_k, res_treino, res_teste


quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN(3, 20, dados_iris['data'], dados_iris['target'])
i=5



# Plotagem dos graficos com os seguintes randoms 1, 5, 20, 550
dados = dados_iris['data']
respostas = dados_iris['target']

plt.rcParams["figure.figsize"] = [12,12]
# RAND 1
legendas = ["Treino", "Teste"]
quantidade = 20
rand = 1
f, axarr = plt.subplots(2,2)
plt.setp(axarr, xticks=np.arange(0,20, step=1))
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN(rand, quantidade, dados, respostas)
axarr[0,0].plot(res_treino)
axarr[0,0].plot(res_teste)
axarr[0,0].grid(True)
axarr[0,0].set_title("RAND 1")
axarr[0,0].legend(legendas)

# RAND 5
rand = 5
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN(rand, quantidade, dados, respostas)
axarr[0,1].plot(res_treino)
axarr[0,1].plot(res_teste)
axarr[0,1].grid(True)
axarr[0,1].set_title("RAND 5")
axarr[0,1].legend(legendas)

# RAND 20

rand = 20
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN(rand, quantidade, dados, respostas)
axarr[1,0].plot(res_treino)
axarr[1,0].plot(res_teste)
axarr[1,0].grid(True)
axarr[1,0].set_title("RAND 20")
axarr[1,0].legend(legendas)

# RAND 550

rand = 550
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN(rand, quantidade, dados, respostas)
axarr[1,1].plot(res_treino)
axarr[1,1].plot(res_teste)
axarr[1,1].grid(True)
axarr[1,1].set_title("RAND 550")
axarr[1,1].legend(legendas)


plt.show()
