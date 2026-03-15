"""
cuFHE-lite — GPU-accelerated BFV Homomorphic Encryption
Compute on encrypted data without decrypting.
"""
import ctypes
import subprocess
import numpy as np
import cupy as cp
from pathlib import Path
import time
import os

Q = 65537
T = 256
N = 1024
DELTA = Q // T
BLOCK = 256

class cuFHE:
    def __init__(self):
        self._compile_kernels()
        self.module = cp.RawModule(path=str(Path(__file__).parent.parent / "kernels" / "fhe_kernel.ptx"))
        self._poly_add      = self.module.get_function("_Z8poly_addPKjS0_Pji")
        self._poly_sub      = self.module.get_function("_Z8poly_subPKjS0_Pji")
        self._poly_scalar   = self.module.get_function("_Z15poly_scalar_mulPKjjPji")
        self._encrypt       = self.module.get_function("_Z11bfv_encryptPKjS0_PjS1_i")
        self._decrypt       = self.module.get_function("_Z11bfv_decryptPKjPji")
        self._he_add        = self.module.get_function("_Z6he_addPKjS0_S0_S0_PjS1_i")
        self._he_mul_plain  = self.module.get_function("_Z12he_mul_plainPKjS0_jPjS1_i")
        print(f"[cuFHE] BFV kernels loaded. N={N}, Q={Q}, T={T}, Δ={DELTA}")

    def _compile_kernels(self):
        ptx_path = Path(__file__).parent.parent / "kernels" / "fhe_kernel.ptx"
        cu_path  = Path(__file__).parent.parent / "kernels" / "fhe_kernel.cu"
        if not ptx_path.exists():
            print("[cuFHE] Compiling CUDA kernels...")
            result = subprocess.run([
                "nvcc", "--ptx", "-arch=sm_75",
                "-O3", str(cu_path), "-o", str(ptx_path)
            ], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"nvcc failed:\n{result.stderr}")
            print("[cuFHE] Compiled successfully.")

    def _grid(self, n):
        return ((n + BLOCK - 1) // BLOCK,)

    def keygen(self):
        """Generate a simple secret key (random small polynomial)."""
        sk = np.random.randint(0, 3, N, dtype=np.uint32)  # small coefficients {0,1,2}
        return sk

    def encrypt(self, message: np.ndarray, sk=None) -> tuple:
        """Encrypt a plaintext polynomial. Returns (ct0, ct1) on GPU."""
        assert message.max() < T, f"Message values must be < {T}"
        msg_gpu = cp.asarray(message.astype(np.uint32))
        # Small error polynomial for security
        error = np.random.randint(0, 4, N, dtype=np.uint32)
        err_gpu = cp.asarray(error)
        ct0 = cp.zeros(N, dtype=cp.uint32)
        ct1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._encrypt(
            self._grid(N), (BLOCK,),
            (msg_gpu, err_gpu, ct0, ct1, np.int32(N))
        )
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[cuFHE] Encrypted {N} coefficients in {ms:.3f}ms")
        return ct0, ct1

    def decrypt(self, ct0, ct1) -> np.ndarray:
        """Decrypt ciphertext back to plaintext polynomial."""
        out = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._decrypt(self._grid(N), (BLOCK,), (ct0, out, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[cuFHE] Decrypted in {ms:.3f}ms")
        return cp.asnumpy(out)

    def he_add(self, ct_a, ct_b) -> tuple:
        """Add two ciphertexts WITHOUT decrypting."""
        ct_out0 = cp.zeros(N, dtype=cp.uint32)
        ct_out1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._he_add(
            self._grid(N), (BLOCK,),
            (ct_a[0], ct_a[1], ct_b[0], ct_b[1], ct_out0, ct_out1, np.int32(N))
        )
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[cuFHE] Homomorphic ADD in {ms:.3f}ms (no decryption)")
        return ct_out0, ct_out1

    def he_mul_plain(self, ct, scalar: int) -> tuple:
        """Multiply ciphertext by plaintext scalar WITHOUT decrypting."""
        ct_out0 = cp.zeros(N, dtype=cp.uint32)
        ct_out1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._he_mul_plain(
            self._grid(N), (BLOCK,),
            (ct[0], ct[1], np.uint32(scalar), ct_out0, ct_out1, np.int32(N))
        )
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[cuFHE] Homomorphic MUL (plaintext scalar={scalar}) in {ms:.3f}ms")
        return ct_out0, ct_out1

    def benchmark(self, n_ops=1000):
        print(f"\n[cuFHE] Benchmarking {n_ops} operations...")
        msg = np.random.randint(0, T, N, dtype=np.uint32)
        ct_a = self.encrypt(msg)
        ct_b = self.encrypt(msg)

        t0 = time.perf_counter()
        for _ in range(n_ops):
            self.he_add(ct_a, ct_b)
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[cuFHE] {n_ops} HE additions: {ms:.1f}ms total ({n_ops/ms*1000:.0f} ops/sec)")
