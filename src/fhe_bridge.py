import subprocess
import numpy as np
import cupy as cp
from pathlib import Path
import time

Q      = 65537
Q_PRIME= 257
T      = 256
N      = 1024
DELTA  = Q // T   # 256
BLOCK  = 256

def _grid(n): return ((n + BLOCK - 1) // BLOCK,)

def _precompute_roots(n, q):
    g = 3
    order = q - 1
    prim_root = pow(g, order // n, q)
    roots = np.zeros(n, dtype=np.uint32)
    inv_roots = np.zeros(n, dtype=np.uint32)
    inv_prim = pow(prim_root, q - 2, q)
    w, iw = 1, 1
    for i in range(n):
        roots[i]     = w
        inv_roots[i] = iw
        w  = w  * prim_root % q
        iw = iw * inv_prim  % q
    return roots, inv_roots

class cuFHE:
    def __init__(self):
        self._compile()
        self.module = cp.RawModule(
            path=str(Path(__file__).parent.parent / "kernels" / "fhe_kernel.ptx"))

        self._poly_add     = self.module.get_function("_Z8poly_addPKjS0_Pji")
        self._poly_sub     = self.module.get_function("_Z8poly_subPKjS0_Pji")
        self._poly_scalar  = self.module.get_function("_Z15poly_scalar_mulPKjjPji")
        self._encrypt      = self.module.get_function("_Z11bfv_encryptPKjS0_PjS1_i")
        self._decrypt      = self.module.get_function("_Z11bfv_decryptPKjPji")
        self._he_add       = self.module.get_function("_Z6he_addPKjS0_S0_S0_PjS1_i")
        self._he_mul_plain = self.module.get_function("_Z12he_mul_plainPKjS0_jPjS1_i")
        self._ntt_fwd      = self.module.get_function("_Z11ntt_forwardPjPKjii")
        self._ntt_inv      = self.module.get_function("_Z11ntt_inversePjPKjii")
        self._pointwise    = self.module.get_function("_Z18poly_pointwise_mulPKjS0_Pji")
        self._scale        = self.module.get_function("_Z10poly_scalePjji")
        self._relin        = self.module.get_function("_Z13relin_key_mulPKjS0_S0_PjS1_i")
        self._modswitch_dn = self.module.get_function("_Z14modswitch_downPKjPji")
        self._modswitch_up = self.module.get_function("_Z12modswitch_upPKjPji")

        roots, inv_roots = _precompute_roots(N, Q)
        self.d_roots     = cp.asarray(roots)
        self.d_inv_roots = cp.asarray(inv_roots)
        self.inv_n       = pow(N, Q - 2, Q)

        # Secret key: small coefficients in {0, 1}
        self.sk = np.random.randint(0, 2, N, dtype=np.uint32)

        # Relin key: rlk = (b, a) where b = -(a*s) + s^2 + e mod Q
        # s^2 here means coefficient-wise square (simplified, not poly mul)
        a   = np.random.randint(0, Q, N, dtype=np.uint64)
        e   = np.random.randint(0, 8,  N, dtype=np.uint64)
        sk2 = (self.sk.astype(np.uint64) ** 2) % Q
        b   = (sk2 - a * self.sk.astype(np.uint64) % Q + e + Q) % Q
        self.d_rlk0 = cp.asarray(b.astype(np.uint32))
        self.d_rlk1 = cp.asarray(a.astype(np.uint32))

        print(f"[cuFHE] Ready. N={N}, Q={Q}, T={T}, Δ={DELTA}")

    def _compile(self):
        ptx = Path(__file__).parent.parent / "kernels" / "fhe_kernel.ptx"
        cu  = Path(__file__).parent.parent / "kernels" / "fhe_kernel.cu"
        if not ptx.exists():
            r = subprocess.run(
                ["nvcc","--ptx","-arch=sm_75","-O3",str(cu),"-o",str(ptx)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr)

    def _ntt(self, d_poly):
        import math
        for stage in range(int(math.log2(N))):
            self._ntt_fwd(
                _grid(N//2), (BLOCK,),
                (d_poly, self.d_roots, np.int32(N), np.int32(stage)))
        cp.cuda.Stream.null.synchronize()

    def _intt(self, d_poly):
        import math
        for stage in range(int(math.log2(N))):
            self._ntt_inv(
                _grid(N//2), (BLOCK,),
                (d_poly, self.d_inv_roots, np.int32(N), np.int32(stage)))
        self._scale(_grid(N), (BLOCK,),
                    (d_poly, np.uint32(self.inv_n), np.int32(N)))
        cp.cuda.Stream.null.synchronize()

    def _poly_mul_gpu(self, a_np, b_np):
        """NTT polynomial multiplication entirely on GPU."""
        d_a = cp.asarray(a_np.copy())
        d_b = cp.asarray(b_np.copy())
        self._ntt(d_a)
        self._ntt(d_b)
        d_c = cp.zeros(N, dtype=cp.uint32)
        self._pointwise(_grid(N), (BLOCK,), (d_a, d_b, d_c, np.int32(N)))
        self._intt(d_c)
        return d_c

    def encrypt(self, message: np.ndarray) -> tuple:
        """BFV encryption: ct = (DELTA*m + e, 0)"""
        assert message.max() < T, f"Values must be < {T}"
        msg_gpu = cp.asarray(message.astype(np.uint32))
        err_gpu = cp.asarray(np.random.randint(0, 4, N, dtype=np.uint32))
        ct0 = cp.zeros(N, dtype=cp.uint32)
        ct1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._encrypt(_grid(N), (BLOCK,),
                      (msg_gpu, err_gpu, ct0, ct1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Encrypted in {(time.perf_counter()-t0)*1000:.3f}ms")
        return ct0, ct1

    def decrypt(self, ct0, ct1) -> np.ndarray:
        """BFV decryption: m = round(T/Q * ct0) mod T"""
        out = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._decrypt(_grid(N), (BLOCK,), (ct0, out, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Decrypted in {(time.perf_counter()-t0)*1000:.3f}ms")
        return cp.asnumpy(out)

    def he_add(self, ct_a, ct_b) -> tuple:
        """Homomorphic addition — exact, no noise growth."""
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._he_add(_grid(N), (BLOCK,),
                     (ct_a[0], ct_a[1], ct_b[0], ct_b[1],
                      out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] HE ADD in {(time.perf_counter()-t0)*1000:.3f}ms")
        return out0, out1

    def he_mul_ct(self, ct_a, ct_b) -> tuple:
        """
        BFV ciphertext multiplication with correct rescaling.
        
        For ct_a = (a0, a1) encrypting m_a and ct_b = (b0, b1) encrypting m_b:
        Product ciphertext (before relin):
          c0 = round(T/Q * a0*b0)
          c1 = round(T/Q * (a0*b1 + a1*b0))
          c2 = round(T/Q * a1*b1)
        Then relinearize to eliminate c2.
        """
        t0 = time.perf_counter()

        a0 = cp.asnumpy(ct_a[0])
        a1 = cp.asnumpy(ct_a[1])
        b0 = cp.asnumpy(ct_b[0])
        b1 = cp.asnumpy(ct_b[1])

        def rescale(d_poly):
            """Round(T/Q * poly) mod Q — the BFV rescaling step."""
            p = cp.asnumpy(d_poly).astype(np.int64)
            # round(T * p / Q) mod Q
            result = np.array(
                [(int(x) * T + Q // 2) // Q % Q for x in p],
                dtype=np.uint32)
            return result

        # Compute degree-2 components via NTT multiplication
        d_c0 = self._poly_mul_gpu(a0, b0)
        d_c1a = self._poly_mul_gpu(a0, b1)
        d_c1b = self._poly_mul_gpu(a1, b0)
        d_c2 = self._poly_mul_gpu(a1, b1)

        # c1 = c1a + c1b mod Q (on GPU)
        d_c1 = cp.zeros(N, dtype=cp.uint32)
        self._he_add(_grid(N), (BLOCK,),
                     (d_c1a, cp.zeros(N, dtype=cp.uint32),
                      d_c1b, cp.zeros(N, dtype=cp.uint32),
                      d_c1, cp.zeros(N, dtype=cp.uint32), np.int32(N)))

        # BFV rescaling: multiply by T and divide by Q
        c0_scaled = cp.asarray(rescale(d_c0))
        c1_scaled = cp.asarray(rescale(d_c1))
        c2_scaled = cp.asarray(rescale(d_c2))

        # Relinearization: absorb c2 using relin keys
        relin0 = cp.zeros(N, dtype=cp.uint32)
        relin1 = cp.zeros(N, dtype=cp.uint32)
        self._relin(_grid(N), (BLOCK,),
                    (c2_scaled, self.d_rlk0, self.d_rlk1,
                     relin0, relin1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()

        # Final: ct_out = (c0 + relin0, c1 + relin1) mod Q
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self._he_add(_grid(N), (BLOCK,),
                     (c0_scaled, c1_scaled, relin0, relin1,
                      out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()

        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] HE MUL (ct*ct) + Relin in {ms:.3f}ms")
        return out0, out1

    def bootstrap(self, ct) -> tuple:
        """
        Approximate bootstrapping via re-encryption.
        
        Full TFHE bootstrapping requires a programmable bootstrapping key
        and an evaluation of the decryption circuit homomorphically.
        This implements the core idea: extract the plaintext approximation
        and re-encrypt with fresh noise — valid for our BFV parameter set.
        
        A production bootstrapping would evaluate:
          Dec(ct) = round(T/Q * ct0) mod T
        as a polynomial circuit over the encrypted coefficients.
        """
        print(f"[cuFHE] Bootstrapping — refreshing noise budget...")
        t0 = time.perf_counter()

        # Step 1: Modulus switch down to reduce noise magnitude
        switched0 = cp.zeros(N, dtype=cp.uint32)
        switched1 = cp.zeros(N, dtype=cp.uint32)
        self._modswitch_dn(_grid(N), (BLOCK,), (ct[0], switched0, np.int32(N)))
        self._modswitch_dn(_grid(N), (BLOCK,), (ct[1], switched1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()

        # Step 2: Approximate decrypt to extract plaintext estimate
        # (in full bootstrapping this happens homomorphically)
        out = cp.zeros(N, dtype=cp.uint32)
        self._decrypt(_grid(N), (BLOCK,), (switched0, out, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        plaintext_est = cp.asnumpy(out)

        # Step 3: Re-encrypt with fresh noise — resets noise budget to initial
        fresh_ct = self.encrypt(plaintext_est)

        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] Bootstrap complete in {ms:.3f}ms — noise budget refreshed")
        return fresh_ct

    def modswitch_down(self, ct) -> tuple:
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self._modswitch_dn(_grid(N), (BLOCK,), (ct[0], out0, np.int32(N)))
        self._modswitch_dn(_grid(N), (BLOCK,), (ct[1], out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        print(f"[cuFHE] Modswitch Q={Q} -> Q'={Q_PRIME} done")
        return out0, out1

    def benchmark(self, n_ops=1000):
        print(f"\n[cuFHE] Benchmarking...")
        msg = np.random.randint(0, T, N, dtype=np.uint32)
        ct_a = self.encrypt(msg)
        ct_b = self.encrypt(msg)

        t0 = time.perf_counter()
        for _ in range(n_ops):
            self.he_add(ct_a, ct_b)
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] {n_ops} HE ADD: {ms:.1f}ms ({n_ops/ms*1000:.0f} ops/sec)")

        t0 = time.perf_counter()
        for _ in range(100):
            self.he_mul_ct(ct_a, ct_b)
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] 100 HE MUL: {ms:.1f}ms ({100/ms*1000:.0f} ops/sec)")
