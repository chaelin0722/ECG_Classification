import numpy as np
from sklearn.svm import SVC
import plotly.graph_objects as go
rs = np.random.RandomState(1234)

# Generate some fake data.
n_samples = 200
# X is the input features by row.
#X = np.zeros((200,3))
#X[:n_samples//2] = rs.multivariate_normal( np.ones(3), np.eye(3), size=n_samples//2)
#X[n_samples//2:] = rs.multivariate_normal(-np.ones(3), np.eye(3), size=n_samples//2)
# Y is the class labels for each row of X.
#Y = np.zeros(n_samples); Y[n_samples//2:] = 1

#print(X.shape)
X_train = [[0.9, 0.16, 0.1], [0.8, 0.14, 0.09], [1.2,0.18,0.11], [1.0, 0.13, 0.12],
           [0.6, 0.2, 0.11], [1.8, 0.3, 0.18], [0.35, 0.08, 0.05], [1.7, 0.09, 0.17],
           [0.4, 0.25, 0.06], [1.5, 0.28, 0.04]]
Y = [0,0,0,0,0,1,1,1,1,1]
X_test = [[1.05, 0.165, 0.095], [0.5, 0.3, 0.07], [0.85, 0.17, 0.14], [0.45, 0.085, 0.096], [0.7, 0.28, 0.085]]
X = np.array(X_train)
Y = np.array(Y)

# Fit the data with an svm
svc = SVC(kernel='linear')
svc.fit(X,Y)

# The equation of the separating plane is given by all x in R^3 such that:
# np.dot(svc.coef_[0], x) + b = 0. We should solve for the last coordinate
# to plot the plane in terms of x and y.
print(svc.intercept_)
print(svc.coef_)
# coef_ : weights assigned to the features when kernel="linear"

# 아래 z 는 식을 정의 한것!
z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]

#tmp = np.linspace(-2,2,51)
#x,y = np.meshgrid(tmp,tmp)
xm, xM = X[:,0].min(), X[:, 0].max()
ym, yM = X[:,1].min(), X[:, 1].max()
x = np.linspace(xm, xM, 10)
y = np.linspace(ym, yM, 10)
x, y =np.meshgrid(x, y)

my_colorscale= [[0, 'rgb(230,230,230)'], [1, 'rgb(230,230,230)']]
fig = go.Figure()
fig.add_surface(x=x, y=y, z=z(x,y), colorscale=my_colorscale, showscale=False, opacity=0.9)
fig.add_scatter3d(x=X[Y==0,0], y=X[Y==0,1], z=X[Y==0,2], mode='markers', marker={'color': 'blue'})
fig.add_scatter3d(x=X[Y==1,0], y=X[Y==1,1], z=X[Y==1,2], mode='markers', marker={'color': 'red'})
fig.update_layout(width=800, height=800)
fig.show()
