import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    N=len(A)
    M=len(A[0])
    transposed_array=np.zeros((M,N))
    for i in range(N):
        for j in range(M):
            transposed_array[j,i]=A[i][j]
    return transposed_array
    