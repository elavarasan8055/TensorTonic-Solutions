import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if seqs==[]:
        return np.zeros((0,0))

    if max_len==None:    
        max_len=np.max([len(seq) for seq in seqs])

       
    return np.array([seq+[pad_value]*(max_len-len(seq)) if len(seq)<=max_len else seq[:max_len] for seq in seqs])
    