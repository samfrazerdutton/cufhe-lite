# cuFHE-lite

GPU-accelerated Homomorphic Encryption using CUDA. Perform addition and multiplication on encrypted data without ever decrypting it.

## Benchmarks (RTX 2060 Mobile)
| Operation | Performance |
|---|---|
| Homomorphic ADD | 6,852 ops/sec |
| Encrypt (N=1024) | ~0.06ms |
| Decrypt (N=1024) | ~0.05ms |

## What it does
- BFV scheme over polynomial ring Z_Q[x] / (x^N + 1)
- Q=65537 (Fermat prime — fastest possible modular reduction)
- All operations run on GPU via custom CUDA kernels
- Python bridge via CuPy

## Stack
CUDA · C++ · CuPy · Python

## Run
```bash
nvcc --ptx -arch=sm_75 -O3 kernels/fhe_kernel.cu -o kernels/fhe_kernel.ptx
python tests/test_fhe.py
```
