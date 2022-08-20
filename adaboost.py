import numpy
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

'''
#generate normal sample data
n_sample_att1 = np.random.uniform(low=0.6,high=1.2, size=100)
n_sample_att2 = np.random.uniform(low=0.12,high=0.2, size=100)
n_sample_att3 = np.random.uniform(low=0.08,high=0.12, size=100)
#print(n_sample_att1)

X_train_n = []
X_train_a = []

for i in range(len(n_sample_att1)):
    X_train_n.append([n_sample_att1[i], n_sample_att2[i], n_sample_att3[i]])

#print(X_train)
# generate abnormal sample data
a_sample_att1_1 = np.random.uniform(high=0.59, low=0.01, size=50)
a_sample_att1_2 = np.random.uniform(high=10, low=1.21, size=50)
a_sample_att2_1 = np.random.uniform(high=0.11, low=0.001, size=50)
a_sample_att2_2 = np.random.uniform(high=1.0, low=0.21,size=50)
a_sample_att3_1 = np.random.uniform(high=0.07, low=0.0001, size=50)
a_sample_att3_2 = np.random.uniform(high=10, low=0.13, size=50)

a_sample_att1 = np.concatenate((a_sample_att1_1, a_sample_att1_2), axis=0)
a_sample_att2 = np.concatenate((a_sample_att2_1, a_sample_att2_2), axis=0)
a_sample_att3 = np.concatenate((a_sample_att3_1, a_sample_att3_2), axis=0)

##print("===== shuffle")
np.random.shuffle(a_sample_att1)
np.random.shuffle(a_sample_att2)
np.random.shuffle(a_sample_att3)

for i in range(len(a_sample_att1)):
    X_train_a.append([a_sample_att1[i], a_sample_att2[i], a_sample_att3[i]])


X_train = np.concatenate((X_train_a, X_train_n), axis=0)

#print(X_train.shape)
#print(X_train)
Y_n=np.zeros(100)
Y_a=np.ones(100)
Y = np.concatenate((Y_a, Y_n), axis=0)
'''

X_train = [[0.9, 0.16, 0.1], [0.8, 0.14, 0.09], [1.2,0.18,0.11], [1.0, 0.13, 0.12],
           [0.6, 0.2, 0.11], [1.8, 0.3, 0.18], [0.35, 0.08, 0.05], [1.7, 0.09, 0.17],
           [0.4, 0.25, 0.06], [1.5, 0.28, 0.04]]
Y = [0,0,0,0,0,1,1,1,1,1]

X_test = [[1.05, 0.165, 0.095], [0.5, 0.3, 0.07], [0.85, 0.17, 0.14], [0.45, 0.085, 0.096], [0.7, 0.28, 0.085]]

Y_test = [0,1,1,1,1]

X = X_train
X = np.array(X_train)
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

clf = AdaBoostClassifier()
clf.fit(X, Y)
preds = clf.predict(X_test)

print("results : ",preds)
