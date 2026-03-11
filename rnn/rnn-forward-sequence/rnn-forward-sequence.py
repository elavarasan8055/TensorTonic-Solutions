import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    batch=X.shape[0]
    T=X.shape[1]
    D=X.shape[2]
    h_dim=h_0.shape[1]
    h_all=np.zeros((batch,T,h_dim))
    for i in range(T):
        h_t=np.tanh(np.dot(X[:,i,:],W_xh.T)+np.dot(h_0,W_hh.T)+np.reshape(-1,1).T)
        h_0=h_t
        h_all[:,i,:]=h_t
    return h_all,h_all[:,-1,:]