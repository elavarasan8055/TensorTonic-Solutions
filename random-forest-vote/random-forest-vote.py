import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions=np.array(predictions)

    arr=[np.unique(predictions[:,i],return_counts=True) for i in range(predictions.shape[1])]


    return [label[0][int(label[1].argmax())] for label in arr]