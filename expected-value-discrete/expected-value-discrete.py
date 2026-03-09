import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x_array=np.array(x)
    p_array=np.array(p)
    if np.sum(p_array) !=1:
        raise ValueError
    return np.dot(x_array.T,p)