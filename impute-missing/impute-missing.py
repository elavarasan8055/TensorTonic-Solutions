import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X=np.array(X)
    X=X.astype(float)
    d=X.ndim
    if d==1:
       X=X.reshape(-1,1)
    
    if strategy=="mean":
        filler_Value=np.nanmean(X,axis=0)
    else:
        filler_Value=np.nanmedian(X,axis=0)

    
    X=np.where(np.isnan(X),filler_Value,X)
    X=np.nan_to_num(X,nan=0)
    return np.squeeze(X) if d==1 else X
    