import numpy as np
from math import sqrt as math_sqrt
import pylab as pl
import matplotlib.cm as cm

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

    UE = np.dot(U, E)
    UEVT = np.dot(UE, VT)

    return np.real(UEVT)


def blockshaped(matrix):
    row_len = len(matrix[0])
    col_len = len(matrix)

    n = 16
    # print('n', n)
    if row_len % n != 0:
        matrix = matrix[:, :-(row_len % n)]
    if col_len % n != 0:
        matrix = matrix[:col_len - (col_len % n)]
    nrows = row_len // n
    ncols = col_len // n
    h, w = matrix.shape
    return (matrix.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)), n


def shuffle(arr):
    blocks, n = blockshaped(arr)
    row_len = len(blocks)
    col_len = len(blocks[0]) * len(blocks[0][0])
    must_fill = np.zeros((row_len, col_len))

    for i in xrange(must_fill.shape[0]):
        join = np.resize(blocks[i], (1, col_len))[0]
        must_fill[i, :] = join
    print('Shuffled')
    return must_fill, n


def ssvd(a, r):
    row_len, col_len = len(a[0]), len(a)
    if row_len != col_len:
        min_ = min(col_len, row_len)
        if min_ != col_len:
            k = row_len - min_
            a = a[:, k // 2:min_ + k // 2]
        else:
            k = col_len - min_
            a = a[k // 2:min_ + k // 2]

    X, n = shuffle(a)

    # print("X", X)
    svd_r = svd(X, r)
    print('Calculated svd')
    shape = np.shape(svd_r)
    shape_im = (len(a), len(a[0]))
    block_size = shape_im[1] // n
    block_in_row = shape_im[1] // block_size
    shape_im = tuple([x - x % n for x in shape_im])
    As = np.zeros(shape_im)
    # print(shape_im)
    for i in range(len(svd_r)):
        div_mod = (i // block_in_row, i % block_in_row)
        As[div_mod[0] * block_size: (div_mod[0] + 1) * block_size,
        div_mod[1] * block_size:(div_mod[1] + 1) * block_size] = \
            np.resize(svd_r[i], (block_size, block_size))
    print('Reshuffle')

    return As


def compareSvds(matrix, rank):
    mat = svd(matrix, rank)
    print('Calculated svd')
    mats = ssvd(matrix, rank)

    print('Calculated ssvd')
    pl.imshow(mat, cmap=cm.Greys_r)
    # pl.savefig('my_Lenna.jpg')
    pl.show()

    pl.imshow(mats, cmap=cm.Greys_r)
    pl.show()


img = color.rgb2gray(io.imread('foto/4.jpg'))
pl.imshow(img, cmap=cm.Greys_r)
# pl.savefig('grey_4.jpg')
# l = []
# for i in range(randint(4,20)):
#     st = []
#     for j in range(randint(5,20)):
#         st.append(randint(1, 240))
#     l.append()
# compareSvds(img, 20)
# print(l)
# (shuffle(l))
compareSvds(img, 20)
