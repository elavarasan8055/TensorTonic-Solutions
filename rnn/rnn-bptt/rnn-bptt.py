import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    dtanh_dz=dh_next*(1-h_t**2)
    dht_dwhh=np.dot(dtanh_dz.T,h_prev)
    dht_dh_prev = np.dot(dtanh_dz,W_hh)
    return (dht_dh_prev,dht_dwhh)