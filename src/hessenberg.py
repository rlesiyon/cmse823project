import numpy as np

def hessenberg_form(Ac):

    '''
    Input:
        A: matrix to be orthogonalized
        Modifies A internally to create a R upper triangular matrix.
    Return:
        Q: orthonormal matrix
        R: Upper triangular matrix
    '''

    r, m = Ac.shape
    A = np.copy(Ac)

    for k in range(m-2):

        x = A[k+1:m,k]

        e1= np.zeros(len(x)); e1[0] = 1
        vk = np.sign(x[0])*np.linalg.norm(x,2)*e1 + x
        vk = vk/np.linalg.norm(vk,2)


        vkvk = np.outer(vk, vk)

        ## left multiplication of A with householder reflector
        A[k+1:m, k:m] = A[k+1:m, k:m] - (2*vkvk@A[k+1:m,k:m])

        ## right multiplication of A with householder reflector
        A[0:m, k+1:m] = A[0:m, k+1:m] - 2*(A[0:m, k+1:m]@vkvk)

    return A
