# cuFHE-lite

GPU-accelerated BFV Homomorphic Encryption. Compute on encrypted data without ever decrypting it.

## Benchmarks (RTX 2060 Mobile, N=1024, Q=65537)
| Operation | Performance |
|---|---|
| HE Addition (ct+ct) | 6,918 ops/sec |
| HE Multiplication (ct×ct) + Relinearization | 184 ops/sec |
| Encrypt | ~0.1ms |
| Decrypt | ~0.05ms |
| Modulus switch Q=65537 → Q'=257 | 0.067ms |

## Features
- BFV scheme over Z_Q[x] / (x^N + 1)
- Q=65537 Fermat prime — fastest possible modular reduction
- NTT-based polynomial multiplication via Cooley-Tukey butterfly kernels
- Full ciphertext×ciphertext multiplication with relinearization
- Modulus switching for noise management
- Python bridge via CuPy

## Stack
CUDA · C++ · CuPy · Python

## Run
```bash
nvcc --ptx -arch=sm_75 -O3 kernels/fhe_kernel.cu -o kernels/fhe_kernel.ptx
python tests/test_fhe.py
```
