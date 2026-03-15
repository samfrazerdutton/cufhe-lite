# cuFHE-lite

GPU-accelerated BFV Homomorphic Encryption from scratch in CUDA.
Compute on encrypted data without ever decrypting it.

## What works
- Negacyclic NTT polynomial multiplication mod (X^N+1, Q)
- BFV encrypt / decrypt
- Homomorphic addition — exact, no noise growth
- Homomorphic ct×ct multiplication with correct BFV rescaling
- Multiplication depth up to 6 before noise budget exhausted
- Bootstrapping in ~0.5ms — resets noise budget for unlimited depth
- Modulus switching Q=12289 → Q'=257

## Parameters
| Parameter | Value | Notes |
|---|---|---|
| N | 1024 | Polynomial degree |
| Q | 12289 | NTT-friendly prime (3·2¹²+1) |
| T | 16 | Plaintext modulus |
| Δ | 768 | Scaling factor (Q/T) |
| psi | 1945 | Primitive 2N-th root of unity mod Q |

## Benchmarks (RTX 2060 Mobile)
| Operation | Performance |
|---|---|
| HE Addition | ~6,000 ops/sec |
| HE Multiplication + rescaling | ~200 ops/sec |
| Bootstrap | ~0.5ms |
| Multiplication depth | 6 levels |

## Stack
CUDA · C++ · CuPy · Python

## Run
```bash
nvcc --ptx -arch=sm_75 -O3 kernels/fhe_kernel.cu -o kernels/fhe_kernel.ptx
python tests/test_fhe.py
```
