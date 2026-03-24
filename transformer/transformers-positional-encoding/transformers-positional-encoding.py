import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    positional_encoding = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(d_model // 2 + d_model % 2):
            positional_encoding[pos][i * 2] = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i * 2 + 1 < d_model: 
                positional_encoding[pos][i * 2 + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))

    return positional_encoding