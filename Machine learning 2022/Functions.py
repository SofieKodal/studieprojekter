import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def multinomal_two(y, W):
    y_hat = np.matmul(np.array(y), W)
    P = [np.exp(el) / (1 + sum(np.exp(y_hat))) for el in y_hat]
    P.append(1 / (1 + sum(np.exp(y_hat))))
    # returns the probabilities and the max class non-zero index
    return P, np.argmax(P) + 1


def one_it_adaboost(actual, predicted):
    # Algorithm 7 p. 294, how weights are updated
    delta = np.array([ac == predicted[i] for i, ac in enumerate(actual)])
    w0 = np.ones(len(delta)) / len(delta)
    epsilon = sum(w0 * (1 - delta))
    alpha = np.log((1 - epsilon) / epsilon) / 2

    for i, el in enumerate(w0):
        if delta[i]:
            w0[i] = el * np.exp(-alpha)
        else:
            w0[i] = el * np.exp(alpha)

    w0 = w0 / sum(w0)

    return w0


def jarccard_sim_cluster(observation_matrix):
    nq = np.sum(observation_matrix, axis=0)
    nz = np.sum(observation_matrix, axis=1)
    N = sum(nz)
    S, D, t1, t2 = 0, 0, 0, 0
    for i, el in enumerate(nq):
        t1 += el * (el - 1) / 2
        t2 += nz[i] * (nz[i] - 1) / 2
    for row in observation_matrix:
        for n in row:
            S += n * (n - 1) / 2
    n_term = N * (N - 1) / 2
    D = S + n_term - t1 - t2

    return S / (n_term - D)


def ROC_plot(y_hat, y, title=''):
    import matplotlib.pyplot as plt
    # Given the predictions and the true values
    y_hat.sort()
    y = y[y_hat.argsort()]
    TPR, FPR = [], []
    n = len(y)
    for i, thresh in enumerate(y_hat):
        TPR.append(sum(y[i:] == 1) / sum(y == 1))
        FPR.append(sum(y[i:] == 0) / sum(y == 0))
    TPR.append(0)
    FPR.append(0)
    plt.plot(FPR, TPR)
    plt.title(title)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    return TPR, FPR


def rand_similarity(n):
    # Given counting matrix, calculate the rand index
    N = np.sum(n)
    S = 0  # where they agree
    for el in n.flatten():
        S += el * (el - 1) / 2
    nq = np.sum(n, axis=0)
    nz = np.sum(n, axis=1)
    D = N * (N - 1) / 2 - sum([el * (el - 1) / 2 for el in nz]) - sum([el * (el - 1) / 2 for el in nq]) + S
    return (S + D) / (N * (N - 1) / 2)


def Jaccard_Index(k1, k2):
    S = 0
    D = 0
    N = len(k1)
    for i in range(N):
        for j in range(i, N):
            if k1[i] == k1[j] and k1[j] == k2[j]:
                S += 1
            elif k1[i] != k1[j] and k1[j] != k2[j]:
                D += 1

    J = S / ((N * (N - 1) / 2) - D)
    return J


def purity_gain(Nr, N1, N2):
    Ir = 1 - max(Nr) / sum(Nr)
    I1 = 1 - max(N1) / sum(N1)
    I2 = 1 - max(N2) / sum(N2)
    return Ir - (sum(N1) * I1 + sum(N2) * I2) / sum(Nr)


def Rand_Index(k1, k2):
    S = 0
    D = 0
    N = len(k1)
    for i in range(N):
        for j in range(i + 1, N):
            if k1[i] == k1[j] and k2[i] == k2[j]:
                S += 1
            elif k1[i] != k1[j] and k2[i] != k2[j]:
                D += 1

    R = (S + D) / ((N * (N - 1) / 2))
    return R


def ard2(d1, d2, d3):
    K = 2
    density = 1 / (1 / K * sum(d1))
    densities = [1 / (1 / K * sum(d2)), 1 / (1 / K * sum(d3))]
    ard = density / (1 / K * sum(densities))
    return ard
