import numpy as np
def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    series=np.array(series)
    mean=np.mean(series)
    covariance=[np.sum((series[i:]-mean)*(series[0:len(series)-i]-mean)) for i in range(max_lag+1)]
    variance=np.sum((series-mean)**2)
    return [1 if lag==0 else 0 if variance==0 else cov/variance for lag,cov in enumerate(covariance)]