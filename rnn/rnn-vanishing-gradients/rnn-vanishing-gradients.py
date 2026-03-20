import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    spectral_norm = np.linalg.norm(W_hh,ord=2)
    start_val=1
    output=[1]
    
    for i in range(T-1):
        start_val=start_val*spectral_norm
        output.append(start_val)
    return output