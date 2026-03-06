import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m=np.array(m)
    param=np.array(param)
    grad=np.array(grad)
    v=np.array(v)
    
    m_updated=beta1*m+(1-beta1)*grad 
    mt=m_updated/(1-beta1**t)
    v_updated=beta2*v + (1-beta2)*grad**2
    vt=v_updated/(1-beta2**t)
    param_new= param - lr*(mt/(np.sqrt(vt)+eps))
    return (param_new,m_updated,v_updated)

    