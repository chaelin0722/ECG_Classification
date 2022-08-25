import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
##https://xavierbourretsicotte.github.io/AdaBoost.html
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
# x1 = rr, x2 = pr, x3 = st
## rr, st

X_train = [[0.9,  0.1], [0.8, 0.09], [1.2,0.11], [1.0, 0.12],
           [0.6, 0.11], [1.8, 0.18], [0.35, 0.05], [1.7, 0.17],
           [0.4, 0.06], [1.5, 0.04]]

'''
## pr, st

X_train = [[0.16, 0.1], [0.14, 0.09], [0.18,0.11], [0.13, 0.12],
           [0.2, 0.11], [0.3, 0.18], [0.08, 0.05], [0.09, 0.17],
           [0.25, 0.06], [0.28, 0.04]]
'''
'''

## rr, pr
X_train = [[0.9, 0.1], [0.8, 0.14], [1.2,0.18], [1.0, 0.13],
           [0.6, 0.2], [1.8, 0.3], [0.35, 0.08], [1.7, 0.09],
           [0.4, 0.25], [1.5, 0.28]]
'''
Y = [0,0,0,0,0,1,1,1,1,1]
'''
X_train = [[0.9, 0.16, 0.1], [0.8, 0.14, 0.09], [1.2,0.18,0.11], [1.0, 0.13, 0.12],
           [0.6, 0.2, 0.11], [1.8, 0.3, 0.18], [0.35, 0.08, 0.05], [1.7, 0.09, 0.17],
           [0.4, 0.25, 0.06], [1.5, 0.28, 0.04]]
Y = [0,0,0,0,0,1,1,1,1,1]
'''

data = pd.DataFrame({'RR':[0.9, 0.8, 1.2, 1.0, 0.6, 1.8, 0.35, 1.7, 0.4, 1.5] ,
                     'PR':[0.1, 0.14, 0.18, 0.13, 0.2, 0.3, 0.08, 0.09, 0.25, 0.28],
                     'class':['normal','normal','normal','normal','normal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal']})


# X_test = [[1.05, 0.165, 0.095], [0.5, 0.3, 0.07], [0.85, 0.17, 0.14], [0.45, 0.085, 0.096], [0.7, 0.28, 0.085]]

X_test = [[1.05, 0.095], [0.5, 0.07], [0.85, 0.14], [0.45, 0.096], [0.7, 0.085]]

X_test_n =[[1.05, 0.095]]
X_test_ab = [[0.5, 0.07], [0.85, 0.14], [0.45, 0.096], [0.7, 0.085]]
Y_test = [0,1,1,1,1]

#X = X_train
X = np.array(X_train)
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test_n = np.array(X_test_n)
X_test_ab = np.array(X_test_ab)


def plot_decision_boundary(classifier, X, y, N=10, scatter_weights=np.ones(len(Y)), ax=None):
    '''Utility function to plot decision boundary and scatter plot of data'''
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))

    # Check what methods are available
    if hasattr(classifier, "decision_function"):
        zz = np.array([classifier.decision_function(np.array([xi, yi]).reshape(1, -1)) for xi, yi in
                       zip(np.ravel(xx), np.ravel(yy))])
    elif hasattr(classifier, "predict_proba"):
        zz = np.array([classifier.predict_proba(np.array([xi, yi]).reshape(1, -1))[:, 1] for xi, yi in
                       zip(np.ravel(xx), np.ravel(yy))])
    else:
        zz = np.array([classifier(np.array([xi, yi]).reshape(1, -1)) for xi, yi in zip(np.ravel(xx), np.ravel(yy))])

    # reshape result and plot
    Z = zz.reshape(xx.shape)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Get current axis and plot
    if ax is None:
        ax = plt.gca()
    ax.contourf(xx, yy, Z, 2, cmap='RdBu', alpha=.5)
    ax.contour(xx, yy, Z, 2, cmap='RdBu')
    #ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, s=scatter_weights * 40)
    ##
    ax.scatter(X_test_n[:, 0], X_test_n[:, 1], c='red')
    ax.scatter(X_test_ab[:, 0], X_test_ab[:, 1], c='blue')
    ax.set_xlabel('$RR$')
    ax.set_ylabel('$ST$')



#for i in range(200):
#n_estimators = i + 1

base_estimator = DecisionTreeClassifier(max_depth=1, random_state=0)
adaboost = AdaBoostClassifier(base_estimator=base_estimator,
                              n_estimators=50, algorithm="SAMME",
                              random_state=0)

adaboost.fit(X,Y)
preds = adaboost.predict(X_test)
score = adaboost.score(X_test, Y_test)

#print("predict : ",preds)
#print("scores : ",score)




clf = AdaBoostClassifier()
clf.fit(X, Y)


score = clf.score(X_test,Y_test)
preds = clf.predict(X_test)

print("score", score)
print("preds", preds)


plot_decision_boundary(clf, X, Y, N = 50)#, weights)
plt.show()
