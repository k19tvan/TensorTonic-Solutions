import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    e = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2 + d_model % 2):
            angle = pos / (base ** (2 * i / d_model))
            e[pos][i * 2] = np.sin(angle)
            if (i * 2 + 1 < d_model): e[pos][i * 2 + 1] = np.cos(angle) 

    return e