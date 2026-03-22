import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # X (N, d) ; y (1, N)
    N, d = X.shape
    X = np.array(X); y = np.array(y)
    
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X @ w.T + b)
        g_w = X.T @ (p - y) 
        g_b = np.sum((p - y))
        
        w = w - lr * g_w
        b = b - lr * g_b

    return w, b
