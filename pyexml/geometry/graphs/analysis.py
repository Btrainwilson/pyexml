from scipy import linalg
import numpy as np

def Proj_k_distance(X, Y, k):
    X_val, X_vec = eigen_decomp(X)
    Y_val, Y_vec = eigen_decomp(Y)

    X_proj = np.zeros(X_vec.shape)
    Y_proj = np.zeros(Y_vec.shape)

    X_proj[:, :k] = X_vec[:, -k:]
    Y_proj[:, :k] = Y_vec[:, -k:]

    P_X = X_proj @ X_proj.T
    P_Y = Y_proj @ Y_proj.T

    return abs((np.trace( X_proj @ X_proj.T ) - np.trace(X_proj @ Y_proj.T)))

def eigen_decomp(L):
    #L - Weighted Graph Laplacian
    #e_vec is matrix where columns are eigenvectors
    e_val, e_vec = np.linalg.eig(L)
    idx = np.abs(e_val).argsort()

    return e_val[idx], e_vec[:, idx]

def Kruglov_distance(X, Y):
    X_val, X_vec = eigen_decomp(X)
    Y_val, Y_vec = eigen_decomp(Y)

    dist = 0

    #Don't use first eigenvector
    for i in range(1, X.shape[1]):
        dist += krug_dist(np.squeeze(X_vec[:, i]), np.squeeze(Y_vec[:, i]))

    return dist / ( X.shape[1] - 1)


def krug_dist(x, y):

    del_H = []
    t = []

    x = np.sort(x)
    y = np.sort(y)

    i = 0
    j = 0

    while (i != len(x)) or (j != len(y)):

        del_H.append(float(abs(i - j)) / len(x))

        if (i == len(x)):
            t.append(y[j])
            j += 1

        elif (j == len(y)):
            t.append(x[i])
            i += 1

        elif x[i] < y[j]:
            t.append(x[i])
            i += 1

        else:
            t.append(y[j])
            j += 1

    sum = 0
    for k in range(len(del_H) - 1):
        sum += del_H[k] * (t[k + 1] - t[k])

    return sum

if __name__ == "__main__":
    X = np.array([[1, -1, 0, 0],[-1, 3, -1, -1],[0, -1, 1, 0],[0, 1, 0, -1]])
    Y = np.array([[1, -1, 0, 0],[-1, 2, -1, 0],[0, -1, 2, -1],[0, 0, 1, -1]])

    print(Kruglov_distance(X, Y))
