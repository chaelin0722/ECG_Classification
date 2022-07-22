from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#X_train_n = [[0.9, 0.16, 0.1], [0.8, 0.14, 0.09], [1.2,0.18,0.11], [1.0, 0.13, 0.12], [0.6, 0.2, 0.11]]
#X_train_an = [[1.8, 0.3, 0.18], [0.35, 0.08, 0.05], [1.7, 0.09, 0.17], [0.4, 0.25, 0.06], [1.5, 0.28, 0.04]] 
X_train = [[0.9, 0.16, 0.1], [0.8, 0.14, 0.09], [1.2,0.18,0.11], [1.0, 0.13, 0.12], [0.6, 0.2, 0.11], [1.8, 0.3, 0.18], [0.35, 0.08, 0.05], [1.7, 0.09, 0.17], [0.4, 0.25, 0.06], [1.5, 0.28, 0.04]] 
y = [0,0,0,0,0,1,1,1,1,1]
X_test = [[1.05, 0.165, 0.95], [0.5, 0.3, 0.07], [0.85, 0.17, 0.14], [0.45, 0.085, 0.096], [0.7, 0.28, 0.085]]
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y).predict(X_test)
print(y_pred)
