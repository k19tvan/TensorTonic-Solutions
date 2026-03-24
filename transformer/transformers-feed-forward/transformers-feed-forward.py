import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """

    z1 = np.maximum(x @ W1 + b1, 0) # 2, 10, 256
    z2 = z1 @ W2 + b2 

    return z2
    