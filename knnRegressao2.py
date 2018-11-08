# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:17:57 2018

@author: Luan
"""


import pandas as pd
import numpy as np
#Fazer os graficos
import matplotlib.pyplot as plt
#para dividir os dados entre teste e treino
from sklearn.model_selection import train_test_split
#para regressao
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

# n_samples = numero de amostras
# n_features = numero de atributos
# noise = tirando a linearidade
# random_state = conjunto de dados
X, Y = make_regression(n_samples=50, n_features=1, noise=30, random_state=5)

# grafico de pontos
plt.scatter(X,Y)
plt.show()


# Função que retorna os valores
def retornaResultadosModeloKNN_regressao(random_state, quantidade, dados, respostas):
    # efetua a divisão dos dados entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(dados, respostas, random_state = random_state)
    #vetores de armazenamento dos resultados de testes e dos treinos
    quantidade_k = range(1, quantidade + 1)
    res_teste = []
    res_treino = []
    
    #loop das classificações
    for i in quantidade_k:
        knn = KNeighborsRegressor(n_neighbors = i)
        knn.fit(X_train, y_train)
        
        #colocando os resultados nos vetores
        res_treino.append(knn.score(X_train, y_train))
        res_teste.append(knn.score(X_test, y_test))
    
    return quantidade_k, res_treino, res_teste


legendas = ["Linha de predição", "Treino", "Teste"]
figura, eixos = plt.subplots(1, 3, figsize=(15,5))
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1)


#código para criar uma linha de predição dos dados de treino, cria 1000 valores de -3 até 3
linha = np.linspace(-3, 3, 1000).reshape(-1,1)

for n_neighbors, ex in zip([1,3,9], eixos):
    reg = KNeighborsRegressor(n_neighbors = n_neighbors)
    reg.fit(X_train, y_train)
    ex.plot(linha, reg.predict(linha))
    ex.plot(X_train, y_train, '^', markersize=5)
    ex.plot(X_test, y_test, 'v', markersize=8)    
    ex.set_title("{} neighbors\n Treino: {:.2f} - Teste: {:.2f}".format(n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    
eixos[0].legend(legendas)
plt.plot()
          
