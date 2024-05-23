import torch

# Initialize two random 3x3 matrices on the GPU
A = torch.rand((3, 3), device='cuda')
B = torch.rand((3, 3), device='cuda')
torch.cuda.profiler.cudart().cudaProfilerStart()
# Start the NVTX range for profiling

# Perform element-wise addition
C = torch.add(A, B)

# End the NVTX range for profiling

torch.cuda.profiler.cudart().cudaProfilerStop()

# Print the matrices
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C (A + B):\n", C)

