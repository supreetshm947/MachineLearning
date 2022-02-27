import numpy as np;
import cvxopt
import matplotlib.pyplot as plt


def svmlin(X, t, C):
    N = X.shape[0]
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H[i][j] = t[i] * t[j] * np.dot(X[i], X[j])
    # negative of function to minimise the dual form
    q = (-1) * np.ones(N)
    G = np.vstack([-np.eye(N), np.eye(N)])
    LB = np.zeros(N)
    UB = C * np.ones(N)
    h = np.hstack([-LB, UB])
    A = np.reshape(t, (1, N)).astype(np.double)
    b = np.double(0)

    sol = cvxopt.solvers.qp(P=cvxopt.matrix(H), q=cvxopt.matrix(q), G=cvxopt.matrix(G),
                            h=cvxopt.matrix(h), A=cvxopt.matrix(A), b=cvxopt.matrix(b))
    alpha = np.array(sol['x']).reshape(-1)

    sv = np.where(alpha > 1e-6, True, False)

    w = (alpha[sv] * t[sv]).dot(X[sv])
    b = np.mean(t[sv] - w.dot(X[sv].T))

    decision = X.dot(w) + b
    return w, b, sv, decision


def plot(X, Y, w, b, sv):
    xmin = np.min(X[:, 0])
    xmax = np.max(X[:, 0])
    ymin = np.min(X[:, 1])
    ymax = np.max(X[:, 1])
    x = np.arange(xmin, xmax, 0.0001)
    y = -(w[0] * x + b) / w[1]
    pos = -(w[0] * x + b + 1) / w[1]
    neg = -(w[0] * x + b - 1) / w[1]

    plt.axis([xmin, xmax, ymin, ymax])
    plt.plot(x, y, "k-")
    plt.plot(x, pos, "b-")
    plt.plot(x, neg, "r-")

    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],c="r", marker=(8,2,0), linewidth=5)
    plt.scatter(X[Y==-1][:,0],X[Y==-1][:,1],c="b",marker=(8,2,0), linewidth=5)
    plt.scatter(X[sv][:,0], X[sv][:,1], marker="o", linewidth=1.5, facecolor="none", edgecolors="limegreen")
    

X, Y = np.loadtxt("lc_train_data.dat"), np.loadtxt("lc_train_label.dat")
C = 1000
w, b, sv, decision = svmlin(X, Y, C)
plot(X, Y, w, b, sv)
plt.savefig("SVM_train.png")
plt.clf()

#test
X_test, Y_test = np.loadtxt("lc_test_data.dat"), np.loadtxt("lc_test_label.dat")
dec_test = np.sign(X_test.dot(w)+b)

xmin = np.min(X_test[:, 0])
xmax = np.max(X_test[:, 0])
ymin = np.min(X_test[:, 1])
ymax = np.max(X_test[:, 1])
x = np.arange(xmin, xmax, 0.0001)
y = -(w[0] * x + b) / w[1]
pos = -(w[0] * x + b + 1) / w[1]
neg = -(w[0] * x + b - 1) / w[1]


plt.axis([xmin, xmax, ymin, ymax])
plt.plot(x, y, "k-")
plt.plot(x, pos, "b-")
plt.plot(x, neg, "r-")


plt.scatter(X_test[Y_test==1][:,0],X_test[Y_test==1][:,1],c="r", marker=(8,2,0), linewidth=5)
plt.scatter(X_test[Y_test==-1][:,0],X_test[Y_test==-1][:,1],c="b",marker=(8,2,0), linewidth=5)
plt.savefig("SVM_test.png")

