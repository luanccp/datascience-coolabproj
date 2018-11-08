from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()

# Para imprimir a descricao completa do dataset:
# print(cancer['DESCR'])

# 569 data points with 30 features
cancer['data'].shape

X = cancer['data']
y = cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Vamos dividir nossos dados em conjuntos de treinamento e testes
X_train.shape
X_test.shape

# Dimensionar os dados atravez de uma normalização, utilizamos a StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##################################### TREINAMENTO #####################################
from sklearn.neural_network import MLPClassifier
# Definição do modelo: 3 camadas com 3 neuronios cada
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

#Agora que o modelo foi definido, podemos ajustar os dados de treinamento ao nosso modelo
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

##################################### EXIBIR #####################################
# Agora podemos usar as métricas incorporadas do SciKit-Learn, como um relatório de classificação e uma matriz de confusão para avaliar o desempenho do nosso modelo
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))



# Se quiser saber os pesos e biases
# mlp.coefs_
# mlp.intercepts_
# conferindo o quantidade de neuronios
len(mlp.coefs_[0])
