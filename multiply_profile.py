import torch

# Start NVTX range for profiling
torch.cuda.nvtx.range_push("start")

# Create test tensors
A = torch.rand((3, 3), device='cuda')
B = torch.rand((3, 3), device='cuda')

# Perform matrix multiplication
C = torch.matmul(A, B)

# End NVTX range
torch.cuda.nvtx.range_pop()

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C (A @ B):\n", C)

