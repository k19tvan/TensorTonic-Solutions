import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    # x_t : (B, n_dim); h_prev : (B, hidden_dim); W_xh : (n_dim, hidden_dim); W_hh : (hidden_dim, hidden_dim)
    # assert 1 == 2, f"{x_t.shape}, {h_prev.shape}, {W_xh.shape}, {W_hh.shape}"
    return np.tanh(h_prev @ W_hh + x_t @ W_xh.T + b_h) 