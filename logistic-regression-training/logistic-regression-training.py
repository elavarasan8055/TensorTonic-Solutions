import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    
    b=np.random.randn()
    x=np.array(X)
    y=np.array(y).reshape(-1,1)
    w=np.random.randn(1,x.shape[-1])
    for step in range(steps):
        yi=_sigmoid(np.dot(x,w.T)+b)
        L=np.sum(-1/len(X) * (y*np.log(yi) + (1-y)*np.log(1-yi)))
        
        dl_dw= (1/len(x))*np.dot((yi-y).T,x)
        dl_db=(1/len(x))*np.sum(yi-y)
        w=w-lr*dl_dw
        b=b-lr*dl_db
        
    return (w[0],b)