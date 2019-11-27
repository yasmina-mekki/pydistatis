import numpy as np
import pandas as pd


def simulate_data(N=10, cst=0.6, dim_r=3, dim_c=3):
    simulated_data = {}
    
    for i in list(range(N)):
        A_star = np.zeros((dim_r, dim_c))
        A_star.fill(cst)
        np.fill_diagonal(A_star, 1)
        simulated_data[i] = A_star
        cst += 0.1
    return simulated_data


  

def compute_compromise_factor_space(compromise_matrix):
    '''
    Estimate the compromise factor space

    Parameters
    ----------
    compromise_matrix : array_like, 2d
                    symetric matrix

    Returns
    -------
    values : array_like, 1d
        eigen values of the compromise matrix
    vectors : array_like, 2d
        eigen vectors of the compromise matrix
    F : array_like, 2d
        compromise factor space
    '''
    
    values, vectors = np.linalg.eigh(compromise_matrix)
    vectors[:, vectors.shape[0]-1] = np.abs(vectors[: ,vectors.shape[0]-1])
    return values, vectors, np.dot(vectors, np.diag(np.sqrt(np.abs(values))))
    
    
def compute_factor_space(Rv_matrix):
    '''
    Estimate the factor space

    Parameters
    ----------
    Rv_matrix : array_like, 2d
            Cosine matrix, symetric matrix

    Returns
    -------
    values : array_like, 1d
        eigen values of the Cosine matrix
    vectors : array_like, 2d
        eigen vectors of the Cosine matrix
    G : array_like, 2d
        factor space
    '''
    
    values, P = np.linalg.eigh(Rv_matrix)
    P[:, P.shape[0]-1] = np.abs(P[: ,P.shape[0]-1])
    return values, P, np.dot(P, np.diag(np.sqrt(np.abs(values))))


def compromise_matrix(data, P):
    '''
    Estimate the compromise Matrix

    Parameters
    ----------
    data : array_like, 2d
            symetric matrix
    P : eigen vectors of the Cosine matrix (Rv matrix)

    Returns
    -------
    compromise_m : array_like, 2d
                compromise matrix
    '''
    
    p1 = P[:, P.shape[0]-1]
    alpha = 1/np.sum(p1) * p1
    i = 0
    compromise_m = np.array([], dtype=np.int64).reshape(0, data[list(data.keys())[0]].shape[1])
    for key in data:
        if i == 0:
            compromise_m = data[key]*alpha[i]
        else:
            compromise_m += data[key]*alpha[i]
        i += 1
    return compromise_m

        
def rvMatrix(data):
    '''
    Estimate the Rv Matrix (Cosine matrix)

    Parameters
    ----------
    data : array_like, 2d
        symetric matrix

    Returns
    -------
    Rv Matrix : array_like, 2d
        symetric matrix
    '''
    
    Rv_matrix = np.zeros(shape=(len(data), len(data)))
    index = 0
    for key in data:
        rv_coeffs = []
        for key_2 in data:
            rv_coeffs.append(rvCoeff(data[key], data[key_2]))
        Rv_matrix[index] = rv_coeffs
        index += 1
    return Rv_matrix

        
def rvCoeff(A, B):
    '''
    Estimate the Rv Coefficient

    Parameters
    ----------
    A : array_like, 2d
        symetric matrix
    B : array_like, 2d
        symetric matrix

    Returns
    -------
    Rv Coefficient : Scalar
    '''
    
    ab = np.sum(A * B)
    aa = np.sum(A * A)
    bb = np.sum(B * B)
    return ab / np.sqrt(aa * bb)


def cov2corr(cov):
    '''
    Convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.
    '''
    
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    return corr
