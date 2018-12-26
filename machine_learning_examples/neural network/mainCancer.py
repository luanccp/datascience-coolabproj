from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()

# 569 data points with 30 features
cancer['data'].shape

X = cancer['data']
y = cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train.shape
X_test.shape


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##################################### TRAINING #####################################
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

###### View ######

from sklearn.metrics import classification_report,confusion_matrix
# confusion matrix
print(confusion_matrix(y_test,predictions))
# model performace
print(classification_report(y_test,predictions))

# to show  bias and weight
# mlp.coefs_
# mlp.intercepts_

# checking neurons
len(mlp.coefs_[0])
