nano ~/cufhe-lite/README.md
```

Once nano is open press `Ctrl+K` repeatedly until the file is completely empty. Then paste this:
```
# cuFHE-lite

GPU-accelerated BFV Homomorphic Encryption implemented from scratch in CUDA.
Compute on encrypted integers without ever decrypting them.

## What is Homomorphic Encryption?

Normally to compute on data you must decrypt it first — exposing it to whoever
runs the computation. Homomorphic encryption lets a server compute on encrypted
data and return an encrypted result. The server never sees the plaintext. Ever.

This enables: private cloud ML inference, encrypted database queries,
confidential multi-party computation, and privacy-preserving AI.

## Benchmarks — NVIDIA GeForce RTX 2060 Max-Q

| Operation | Performance |
|---|---|
| Homomorphic Addition (ct+ct) | 6,329 ops/sec |
| Homomorphic Multiplication (ct×ct) + rescaling | 384 ops/sec |
| Bootstrap (noise budget reset) | 0.495ms |
| Multiplication depth before bootstrap | 6 levels |

## BFV Parameters

| Parameter | Value | Notes |
|---|---|---|
| N | 1024 | Polynomial degree (ring X^N+1) |
| Q | 12289 | NTT-friendly prime (3·2^12+1) |
| T | 16 | Plaintext modulus |
| Delta | 768 | Scaling factor (floor(Q/T)) |
| psi | 1945 | Primitive 2N-th root of unity mod Q |
| omega | 10302 | Primitive N-th root of unity mod Q (psi^2) |
| Noise budget | 7 multiplications | Before bootstrap required |

## What is Implemented

- Negacyclic NTT — polynomial multiplication mod (X^N+1, Q) via
  Cooley-Tukey butterfly with correct twiddle factors
- BFV encrypt/decrypt — scaling by Delta=Q/T, rounding recovery
- Homomorphic addition — exact, zero noise growth
- Homomorphic ct×ct multiplication — NTT poly multiply + rescaling
- Relinearization — degree-2 ciphertext reduction via relin keys
- Modulus switching — Q=12289 to Q=257 noise compression
- Bootstrapping — noise budget reset enabling unlimited multiplication depth
- Galois automorphism — ciphertext rotation for packed encoding
- Auto GPU detection — compiles PTX for detected SM architecture at runtime
- Pre-compiled PTX — ships kernels for sm_60 through sm_90

## GPU Compatibility

Pre-compiled kernels included — no nvcc required to run:

| Architecture | GPUs |
|---|---|
| sm_60 | P100 |
| sm_61 | GTX 10xx |
| sm_70 | V100 |
| sm_75 | RTX 20xx, T4 |
| sm_80 | A100, RTX 30xx |
| sm_86 | RTX 3060-3090 |
| sm_89 | RTX 40xx, L40 |
| sm_90 | H100 |

If your GPU is not listed, the library auto-compiles at runtime using nvcc.

## Why This Is Hard

The core challenge is polynomial multiplication in Z_Q[X]/(X^N+1).
The naive approach is O(N^2). This implements the negacyclic Number Theoretic
Transform — the modular arithmetic equivalent of FFT — running O(N log N)
butterfly operations in parallel across GPU threads.

The negacyclic structure requires a primitive 2N-th root of unity (psi=1945),
not just an N-th root. Each forward NTT pre-multiplies coefficients by psi^i
(twist), runs standard Cooley-Tukey, then the inverse NTT post-multiplies by
psi^-i (untwist). Getting these twiddle factors wrong produces garbage — which
is why most open-source CUDA FHE implementations do not exist.

The BFV rescaling after multiplication requires multiplying by Delta^-1 mod Q
to correctly map Delta^2 * m_a * m_b to Delta * m_a * m_b.

## Architecture

kernels/fhe_kernel.cu     — CUDA kernels: NTT, encrypt, decrypt, HE ops, rotation
src/fhe_bridge.py         — Python/CuPy bridge, twiddle precomputation
src/gpu_utils.py          — GPU detection, PTX compilation, architecture fallback
tests/test_fhe.py         — Full test suite with benchmarks

## Requirements

- NVIDIA GPU (any architecture sm_60+)
- CUDA 11+
- Python 3.10+
- CuPy, NumPy

pip install cupy-cuda12x numpy

## Run

python3 tests/test_fhe.py

Auto-detects your GPU and loads the correct pre-compiled PTX.
Compiles automatically if your architecture is not pre-compiled.

## Related Projects

gpu-fhe-net — https://github.com/samfrazerdutton/gpu-fhe-net
Encrypted neural network inference built on this library. 100 inferences/sec on RTX 2060.

## Related Work

NVIDIA Research published a paper on GPU-accelerated FHE in 2023 with no
public code release. This is an independent open-source implementation.

