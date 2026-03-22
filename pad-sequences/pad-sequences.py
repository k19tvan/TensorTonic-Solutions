import numpy as np
    
def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here

    max_seq = max([len(x) for x in seqs]) if not max_len else max_len
    
    def fix(seq, max_seq, pad_value):
        seq = seq + [pad_value] * max(0, (max_seq - len(seq)))
        if len(seq) >= max_seq: return seq[:max_seq]
        return seq

    return [fix(x, max_seq, pad_value) for x in seqs]