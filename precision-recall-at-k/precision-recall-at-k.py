import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended=np.array(recommended)
    relevant=np.array(relevant)
    first_k=recommended[:k]
    precision_mask=np.isin(first_k,relevant).astype(int)
    recall_mask=np.isin(relevant,first_k).astype(int)
    precision=np.sum(precision_mask)/k
    recall=np.sum(recall_mask)/len(relevant)
    return [float(precision),float(recall)]