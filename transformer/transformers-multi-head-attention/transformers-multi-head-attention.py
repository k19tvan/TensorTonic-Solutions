import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def attention(Q, K, V):
    d_k = K.shape[-1]
    score = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    return softmax(score) @ V
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """             
    # Q : (batch, sequence_len, d_model) ; W_q : (d_model, d_k)
    B, N, d_model = Q.shape
    h = num_heads
    d_k = d_model // h
    
    Q = (Q @ W_q).reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
    K = (K @ W_k).reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
    V = (V @ W_v).reshape(B, N, h, d_k).transpose(0, 2, 1, 3)

    combined_head = attention(Q, K, V).transpose(0, 2, 1, 3).reshape(B, N, d_model)
    output = combined_head @ W_o
    return output