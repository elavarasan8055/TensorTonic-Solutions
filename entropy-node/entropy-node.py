import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y=np.array(y)
    value,cnt=np.unique(y,return_counts=True)
    probablity=cnt/np.sum(cnt)

    return_value=-np.sum(probablity*np.log2(probablity))
    return return_value if return_value>0 else 0.0