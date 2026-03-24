import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = K.shape[2]
    return torch.softmax(Q @ K.transpose(1, 2) / math.sqrt(d_k), -1) @ V

    # 1 x 4 x 64 @ 1 x 64 x 4 -> 1 x 4 x 4 @ 1 x 4 x d_v