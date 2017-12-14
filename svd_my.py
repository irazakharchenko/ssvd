import numpy as np
from math import sqrt as math_sqrt, ceil
import pylab as pl
import matplotlib.cm as cm
from random import randint
from skimage import color, io
from scipy._lib.six import xrange

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
    if type(a) != 'numpy.ndarray':
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
    #print("U", np.shape(U), "\nE", np.shape(E), "\nVT", np.shape(VT))
    UE = np.dot(U, E)
    UEVT = np.dot(UE, VT)
    #print("A", A, "\nmaybe A", np.real(UEVT))
    #pl.imshow(A, cmap=cm.Greys_r)
    #pl.show()
    #pl.imshow(np.real(UEVT), cmap=cm.Greys_r)
    #pl.show()
    return np.real(UEVT)


def blockshaped(matrix):
    row_len = len(matrix[0])
    col_len = len(matrix)
    n = round((row_len*col_len)**(1.0/4))
    #print('n', n)
    if row_len % n != 0:
        matrix = matrix[:,:-(row_len % n) ]
    if col_len % n != 0:
        matrix = matrix[:col_len-(col_len % n)]
    nrows = row_len // n
    ncols = col_len // n
    h, w = matrix.shape
    return (matrix.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)), n


def shuffle(arr):
    blocks, n = blockshaped(arr)
    row_len = len(blocks)
    col_len = len(blocks[0]) * len(blocks[0][0])
    must_fill = np.zeros((row_len, col_len))
    #print("must fill ",must_fill)

    for i in xrange(must_fill.shape[0]):
        join = np.resize(blocks[i], (1, col_len))[0]
        must_fill[i,:] = join
    return must_fill,n


def ssvd(a, r):
    # X[i//n + n(i % n)][j//n + n(j % n)]
    X, n = shuffle(a)
    #print("X", X)
    svd_r = svd(X, r)

    #print(svd_r, np.shape(svd_r), len(a), len(a[0]))
    #print(X[0][:13], a[0][:13])
    #shape = np.shape(svd_r)
    shape =  np.shape(svd_r)
    As = np.zeros((len(a), len(a[0])))



    return As

def compareSvds(matrix, rank):
    mat = svd(matrix, rank)
    print('Calculated svd')
    mats = ssvd(matrix, rank)
    print('Calculated ssvd')
    pl.imshow(mat, cmap=cm.Greys_r)
    pl.savefig('my_Lenna.jpg')
    pl.show()

    pl.imshow(mats, cmap=cm.Greys_r)
    pl.show()


img = color.rgb2gray(io.imread('foto/Lenna.jpg'))
pl.imshow(img, cmap=cm.Greys_r)
pl.savefig('grey_Lenna.jpg')
compareSvds(img, 20)
