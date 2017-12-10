import numpy as np
from math import sqrt


def sorterForEigenValuesAndVectors(val, vect):
    """
    vect need to be transpose
    """
    n = len(val)
    for i in range(n):
        for j in range(i, n-1):
            if val[j] < val[j+1]:
                val[j], val[j+1] = val[j+1], val[j]
                vect[j], vect[j+1] = np.array(vect[j+1]), np.array(vect[j])
    return val, vect

def makeDiagonalFromValues(val, matr_fat):
    n = len(val)
    r = np.zeros((n,matr_fat))
    for i in range(n):
        r[i,i] = val[i]
    return r

def make_U(sigm, matr, vect):

    m = np.shape(matr)[0]
    n = np.shape(vect)[1]
    vect = vect.T
    u = np.zeros((m, n))
    # print(n)
    for i in range(n):
        # print("i", i)
        u[:, i] = (1 / sigm[i]) * np.dot(matr, vect[i])
    return u


def svd(a, r):
    A = np.array(a)
    ATA = np.dot(A.T, A)
    eig_val, eig_vect = np.linalg.eig(ATA)
    # sort values vectors, make diag matrix
    eig_val, eig_vect_t = sorterForEigenValuesAndVectors(eig_val, eig_vect.T)
    # print(eig_val,'\n', eig_vect)
    min_len = min(r, len(eig_val))
    VT = eig_vect_t[:min_len, :]
    eig_val = np.array([x for x in eig_val if x > 1.e-8])[:min_len]
    # print(eig_vect)
    eig_vect_t = eig_vect_t[:len(eig_val), :]
    # print(eig_vect_t)
    eig_vect = eig_vect_t.T
    # print(eig_vect)
    S = np.array([sqrt(x) for x in eig_val])
    E = makeDiagonalFromValues(S, r)
    # print(E, '\n', VT)

    U = make_U(S, A, eig_vect)
    print(U, "\nE", E, "\nVT", VT)
    UEVT = np.dot(np.dot(U, E), VT)
    print("A", A, "\nmaybe A", UEVT)

svd([[3,2,2], [2,3,-2]], 1)