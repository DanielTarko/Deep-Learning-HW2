import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("MPS is available")
else:
    print("MPS is not available")