import numpy as np
from math import sqrt as math_sqrt

import PIL.Image
import pylab as pl
import matplotlib.cm as cm
from random import randint


def sorterForEigenValuesAndVectors(val, vect):
    """
    vect need to be transpose
    """
    n = len(val)
    for i in range(n):
        for j in range(i, n - 1):
            if val[j] < val[j + 1]:
                val[j], val[j + 1] = val[j + 1], val[j]
                vect[j], vect[j + 1] = np.array(vect[j + 1]), np.array(vect[j])
    return val, vect


def makeDiagonalFromValues(val, matr_fat, rank):
    n = min(len(val), rank)
    r = np.zeros((n, matr_fat))
    for i in range(n):
        r[i, i] = val[i]
    return r


def make_U(sigm, matr, vect, rank):
    m = np.shape(matr)[0]
    n = min(rank, np.shape(vect)[1], len(sigm))
    vect = vect.T
    u = np.zeros((m, n))
    # print(n)
    for i in range(n):
        # print("i", i)
        u[:, i] = (1.0 / sigm[i]) * np.dot(matr, vect[i])
    return u


def svd(a, r):
    A = np.array(a)
    ATA = np.dot(A.T, A)
    eig_val, eig_vect = np.linalg.eig(ATA)
    # sort values vectors, make diag matrix
    eig_val, eig_vect_t = sorterForEigenValuesAndVectors(eig_val, eig_vect.T)
    min_len = r
    VT = eig_vect_t[:min_len, :]
    eig_val = np.array([x for x in eig_val if x > 1.e-8])
    eig_vect = (eig_vect_t[:len(eig_val), :]).T
    S = np.array([math_sqrt(x) for x in eig_val])
    E = makeDiagonalFromValues(S, np.shape(VT)[0], min_len)
    U = make_U(S, A, eig_vect, min_len)
    print("U", np.shape(U), "\nE", np.shape(E), "\nVT", np.shape(VT))
    UE = np.dot(U, E)
    UEVT = np.dot(UE, VT)
    print("A", A, "\nmaybe A", np.real(UEVT))
    pl.imshow(A, cmap=cm.Greys_r)
    pl.show()
    pl.imshow(np.real(UEVT), cmap=cm.Greys_r)
    pl.show()


image = PIL.Image.open('foto/Lenna.jpg')
im_grey = image.convert('L')
l = []
for i in range(4):
    li = []
    for j in range(10):
        li.append(randint(0, 40))
    l.append(li)

svd(im_grey, 20)
# svd([[3,2,2],[2,3,-2]], 3)
# svd(l, 2)
