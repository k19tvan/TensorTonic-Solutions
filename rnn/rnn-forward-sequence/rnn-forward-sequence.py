import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """


    # h_0 : (B, hidden_dim), X : (B, T, n_dim), W_xh : (n_dim, hidden_dim), W_hh : (hidden_dim, hidden_dim), b_h (hidden_dim) 
    # assert 1 == 2, f"{X.shape}, {h_0.shape}, {W_xh.shape}, {W_hh.shape}, {b_h.shape}"
    
    # YOUR CODE HERE

    B, T, n_dim = X.shape
    h_t = [h_0] + [0] * T 

    X = X.transpose(1, 0, 2)
     
    for t in range(1, T + 1): h_t[t] = np.tanh(h_t[t - 1] @ W_hh + X[t - 1] @ W_xh.T + b_h) # (B, hidden_dim)

    # (B, hidden_dim) -> (B, T, hidden_dim)
    h_all = np.stack(h_t).transpose(1, 0, 2)
    return h_all[:, 1:, :], h_all[:, -1, :]
        