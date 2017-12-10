import numpy as np
from math import sqrt
import PIL.Image
import pylab as pl
import matplotlib.cm as cm

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
    print(A)
    ATA = np.dot(A.T, A)
    print(ATA)
    eig_val, eig_vect = np.linalg.eig(ATA)
    # sort values vectors, make diag matrix
    eig_val, eig_vect_t = sorterForEigenValuesAndVectors(eig_val, eig_vect.T)
    # print(eig_val,'\n', eig_vect)
    min_len = r
    VT = eig_vect_t[:min_len, :]
    print("\nVT", np.shape(VT))
    eig_val = np.array([x for x in eig_val if abs(x) > 1.e-8])[:min_len]
    # print(eig_vect)
    #eig_vect_t = eig_vect_t[:len(eig_val), :]
    # print(eig_vect_t)
    eig_vect = eig_vect_t[:len(eig_val), :].T
    # print(eig_vect)
    S = np.array([sqrt(abs(x))%256 for x in eig_val])
    E = makeDiagonalFromValues(S, np.shape(VT)[0])
    # print(E, '\n', VT)

    U = make_U(S, A, eig_vect)
    print(np.shape(U), "\nE", np.shape(E), "\nVT", np.shape(VT))
    UEVT = np.dot(np.dot(U, E), VT)
    print("A", A, "\nmaybe A", UEVT)
    pl.imshow(A, cmap=cm.Greys_r)
    pl.show()
    pl.imshow(UEVT, cmap=cm.Greys_r)
    pl.show()

image = PIL.Image.open('foto/Lenna.jpg')
im_grey = image.convert('L')
svd([[3,2,2], [2,3,-2]], 5)