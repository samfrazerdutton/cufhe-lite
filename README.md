# cuFHE-lite

GPU-accelerated BFV Homomorphic Encryption implemented from scratch in CUDA.
Compute on encrypted integers without ever decrypting them.

## What is Homomorphic Encryption?

Normally to compute on data you must decrypt it first — exposing it to whoever
runs the computation. Homomorphic encryption lets a server compute on encrypted
data and return an encrypted result. The server never sees the plaintext. Ever.

This enables: private cloud ML inference, encrypted database queries,
confidential multi-party computation, and privacy-preserving AI.

## Benchmarks — NVIDIA GeForce RTX 2060 Max-Q (SM_75, 6GB VRAM)

| Operation | Performance |
|---|---|
| Homomorphic Addition (ct+ct) | **6,212 ops/sec** |
| Homomorphic Multiplication (ct×ct) + rescaling | **225 ops/sec** |
| Bootstrap (noise budget reset) | **0.435ms** |
| Multiplication depth before bootstrap | **6 levels** |

## BFV Parameters

| Parameter | Value | Notes |
|---|---|---|
| N | 1024 | Polynomial degree (ring X^N+1) |
| Q | 12289 | NTT-friendly prime (3·2¹²+1) |
| T | 16 | Plaintext modulus |
| Δ | 768 | Scaling factor (⌊Q/T⌋) |
| ψ | 1945 | Primitive 2N-th root of unity mod Q |
| ω | 10302 | Primitive N-th root of unity mod Q (ψ²) |
| Noise budget | ~7 multiplications | Before bootstrap required |

## What's implemented

- **Negacyclic NTT** — polynomial multiplication mod (X^N+1, Q) via
  Cooley-Tukey butterfly with correct twiddle factors (ψ-twist pre/post multiply)
- **BFV encrypt/decrypt** — scaling by Δ=Q/T, rounding recovery
- **Homomorphic addition** — exact, zero noise growth
- **Homomorphic ct×ct multiplication** — NTT poly multiply + DELTA_inv rescaling
- **Relinearization** — degree-2 ciphertext reduction via relin keys
- **Modulus switching** — Q=12289 → Q'=257 noise compression
- **Bootstrapping** — noise budget reset enabling unlimited multiplication depth
- **Auto GPU detection** — compiles PTX for detected SM architecture at runtime

## Why this is hard

The core challenge is polynomial multiplication in Z_Q[X]/(X^N+1).
The naive approach is O(N²). This implements the **negacyclic Number Theoretic
Transform** — the modular arithmetic equivalent of FFT — running O(N log N)
butterfly operations in parallel across GPU threads.

The negacyclic structure requires a primitive **2N-th** root of unity (ψ=1945),
not just an N-th root. Each forward NTT pre-multiplies coefficients by ψⁱ
(twist), runs standard Cooley-Tukey, then the inverse NTT post-multiplies by
ψ⁻ⁱ (untwist). Getting these twiddle factors wrong produces garbage — which
is why most open-source CUDA FHE implementations don't exist.

The BFV rescaling after multiplication requires multiplying by Δ⁻¹ mod Q
(not T/Q as naively expected) to correctly map Δ²·m_a·m_b → Δ·m_a·m_b.

## Architecture
```
kernels/fhe_kernel.cu     — CUDA kernels: NTT, encrypt, decrypt, HE ops
src/fhe_bridge.py         — Python/CuPy bridge, twiddle precomputation
tests/test_fhe.py         — Full test suite with benchmarks
```

## Run
```bash
# Auto-compiles PTX for your GPU architecture
python tests/test_fhe.py
```

Requirements: NVIDIA GPU, CUDA toolkit, CuPy

## GPU compatibility

Tested on SM_75 (RTX 20xx). Recompile for other architectures:
```bash
nvcc --ptx -arch=sm_80 -O3 kernels/fhe_kernel.cu -o kernels/fhe_kernel_sm80.ptx  # RTX 30xx / A100
nvcc --ptx -arch=sm_89 -O3 kernels/fhe_kernel.cu -o kernels/fhe_kernel_sm89.ptx  # RTX 40xx
nvcc --ptx -arch=sm_90 -O3 kernels/fhe_kernel.cu -o kernels/fhe_kernel_sm90.ptx  # H100
```

## Related work

NVIDIA Research published a paper on GPU-accelerated FHE in 2023 with no
public code release. This is an independent open-source implementation.
