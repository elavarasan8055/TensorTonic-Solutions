import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        T=X.shape[1]
        h_0=np.zeros((X.shape[0],self.W_hh.shape[0]))
        y_all=np.zeros((X.shape[0],T,self.W_hy.shape[0]))
        for i in range(T):
            h_t=np.tanh(np.dot(X[:,i,:],self.W_xh.T)+np.dot(h_0,self.W_hh)+self.b_h)
            h_0=h_t
            y_all[:,i,:]=np.dot(h_t,self.W_hy.T)+self.b_y

        return (y_all,h_t)