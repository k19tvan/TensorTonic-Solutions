import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    return gamma * (x - mean) / (np.sqrt(var + eps)) + beta

def attention(Q, K, V):
    d_k = Q.shape[-1]
    return softmax(Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)) @ V # (B x h x N x d_k)
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Q : B x N x n_dim
    # W_q : B x n_dim x d_k
    # Q -> Q (B x N x h x d_k) -> Q (B x h x N x d_k)

    B, N, n_dim = Q.shape
    d_k = n_dim // num_heads
    h = num_heads

    Q = (Q @ W_q).reshape(B, N, h, d_k).transpose(0, 2, 1, 3) 
    K = (K @ W_k).reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
    V = (V @ W_v).reshape(B, N, h, d_k).transpose(0, 2, 1, 3)

    combined_head = attention(Q, K, V).transpose(0, 2, 1, 3).reshape(B, N, n_dim)

    return combined_head @ W_o
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """

    # x : B x N x n_dim
    # W1 : n_dim x d_ff
    
    return np.maximum(x @ W1 + b1, 0) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    x = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    output = layer_norm(x + feed_forward(x, W1, b1, W2, b2), gamma2, beta2)

    return output