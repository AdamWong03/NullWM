# compute SVD and get top-k direction placeholder
import torch

def compute_nullspace_basis(E, k=16):
    U, S, Vt = torch.linalg.svd(E, full_matrices=False)
    return Vt[:k]
