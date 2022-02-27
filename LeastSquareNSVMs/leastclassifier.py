import numpy as np
import matplotlib.pyplot as plt;

def leastsquares(x,y):
    n = x.shape[0]
    x = np.insert(x, 0, 1, axis=1)

    weight = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return weight[0], weight[1:]

def plot(x,y,bias, weight):
    xmin = np.min(x[:,0])
    xmax = np.max(x[:,0])
    ymin = np.min(x[:,1])
    ymax = np.max(x[:,1])
    plt.axis([xmin,xmax,ymin,ymax])
    plt.plot(x[y==1][:,0], x[y==1][:,1], c="r",  marker = 'x', linestyle='none', markersize=5)
    plt.plot(x[y==-1][:,0], x[y==-1][:,1], c="b",  marker = 'o', linestyle='none', markersize=5)
    plt.plot([xmin,xmax],[-(weight[0]*xmin+b)/weight[1], -(weight[0]*xmax+b)/weight[1]], c="g")

x, y = np.loadtxt("lc_train_data.dat"), np.loadtxt("lc_train_label.dat")
b, w = leastsquares(x, y)

plot(x,y, b, w)
plt.savefig("least_classifier_train.png")
x_test, y_test = np.loadtxt("lc_test_data.dat"), np.loadtxt("lc_test_label.dat")
plot(x_test,y_test, b, w)
plt.savefig("least_classifier_test.png")