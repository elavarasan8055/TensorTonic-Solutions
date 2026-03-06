import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos_encoding=[[k(j*(1/base**(2*i/d_model))) for i in range(d_model//2) for k in [np.sin,np.cos]] for j in range(seq_len)]
    print(pos_encoding)
    if d_model%2!=0:
        pos_encoding=[pos+[np.sin(idx*(1/base**(2*(d_model//2)/d_model)))] for idx,pos in enumerate(pos_encoding)]

    return np.round(pos_encoding,19)