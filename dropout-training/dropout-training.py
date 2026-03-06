import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here

    x=np.array(x)
    if rng==None:
        rng=np.random.default_rng()
    else:
        rng=np.random.default_rng(rng)
    probablity_mask=rng.random(x.shape)
    print(probablity_mask)
    output_mask=(probablity_mask<(1-p)).astype(int) if p>0 else np.full(x.shape,1)
    print(output_mask)
    scaled_matrix=(1/np.full(x.shape,(1-p))) if p>0 else np.full(x.shape,1)
    return (x*output_mask*scaled_matrix,output_mask*scaled_matrix)

