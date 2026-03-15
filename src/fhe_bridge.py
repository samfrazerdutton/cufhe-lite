import subprocess
import numpy as np
import cupy as cp
from pathlib import Path
import time

Q      = 65537
Q_PRIME= 257
T      = 256
N      = 1024
DELTA  = Q // T
BLOCK  = 256

def _grid(n): return ((n + BLOCK - 1) // BLOCK,)

def _precompute_roots(n, q):
    """Precompute NTT roots of unity for Cooley-Tukey."""
    # Find a primitive root g of q
    # For q=65537, g=3 is a primitive root
    g = 3
    order = q - 1
    # Primitive n-th root of unity
    prim_root = pow(g, order // n, q)
    roots = np.zeros(n, dtype=np.uint32)
    inv_roots = np.zeros(n, dtype=np.uint32)
    inv_prim = pow(prim_root, q - 2, q)  # modular inverse
    w = 1
    iw = 1
    for i in range(n):
        roots[i]     = w
        inv_roots[i] = iw
        w  = w  * prim_root % q
        iw = iw * inv_prim  % q
    return roots, inv_roots

def _gen_relin_keys(sk, n, q):
    """Generate relinearization keys for ciphertext multiplication."""
    # rlk = (-(a*s) + s^2 + e, a) for random a and small error e
    a   = np.random.randint(0, q, n, dtype=np.uint32)
    e   = np.random.randint(0, 4,  n, dtype=np.uint32)
    sk2 = np.array([(int(sk[i])**2) % q for i in range(n)], dtype=np.uint32)
    rlk0 = (sk2.astype(np.int64) - (a.astype(np.int64) * sk.astype(np.int64) % q) + e.astype(np.int64) + q) % q
    rlk1 = a
    return rlk0.astype(np.uint32), rlk1.astype(np.uint32)

class cuFHE:
    def __init__(self):
        self._compile()
        self.module = cp.RawModule(
            path=str(Path(__file__).parent.parent / "kernels" / "fhe_kernel.ptx"))

        # Old kernels
        self._poly_add     = self.module.get_function("_Z8poly_addPKjS0_Pji")
        self._poly_sub     = self.module.get_function("_Z8poly_subPKjS0_Pji")
        self._poly_scalar  = self.module.get_function("_Z15poly_scalar_mulPKjjPji")
        self._encrypt      = self.module.get_function("_Z11bfv_encryptPKjS0_PjS1_i")
        self._decrypt      = self.module.get_function("_Z11bfv_decryptPKjPji")
        self._he_add       = self.module.get_function("_Z6he_addPKjS0_S0_S0_PjS1_i")
        self._he_mul_plain = self.module.get_function("_Z12he_mul_plainPKjS0_jPjS1_i")

        # New kernels
        self._ntt_fwd      = self.module.get_function("_Z11ntt_forwardPjPKjii")
        self._ntt_inv      = self.module.get_function("_Z11ntt_inversePjPKjii")
        self._pointwise    = self.module.get_function("_Z18poly_pointwise_mulPKjS0_Pji")
        self._scale        = self.module.get_function("_Z10poly_scalePjji")
        self._relin        = self.module.get_function("_Z13relin_key_mulPKjS0_S0_PjS1_i")
        self._modswitch_dn = self.module.get_function("_Z14modswitch_downPKjPji")
        self._modswitch_up = self.module.get_function("_Z12modswitch_upPKjPji")

        # Precompute roots of unity on GPU
        roots, inv_roots = _precompute_roots(N, Q)
        self.d_roots     = cp.asarray(roots)
        self.d_inv_roots = cp.asarray(inv_roots)
        self.inv_n       = pow(N, Q - 2, Q)  # N^{-1} mod Q

        # Generate secret key + relin keys
        self.sk    = np.random.randint(0, 3, N, dtype=np.uint32)
        rlk0, rlk1 = _gen_relin_keys(self.sk, N, Q)
        self.d_rlk0 = cp.asarray(rlk0)
        self.d_rlk1 = cp.asarray(rlk1)

        print(f"[cuFHE] All kernels loaded. N={N}, Q={Q}, T={T}, Δ={DELTA}")
        print(f"[cuFHE] NTT roots precomputed. Relin keys generated.")

    def _compile(self):
        ptx = Path(__file__).parent.parent / "kernels" / "fhe_kernel.ptx"
        cu  = Path(__file__).parent.parent / "kernels" / "fhe_kernel.cu"
        if not ptx.exists():
            print("[cuFHE] Compiling...")
            r = subprocess.run(
                ["nvcc","--ptx","-arch=sm_75","-O3",str(cu),"-o",str(ptx)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr)

    # ── NTT helpers ───────────────────────────────────────────────────────────

    def _ntt(self, d_poly):
        """Forward NTT in-place on GPU."""
        import math
        log_n = int(math.log2(N))
        for stage in range(log_n):
            self._ntt_fwd(
                _grid(N // 2), (BLOCK,),
                (d_poly, self.d_roots, np.int32(N), np.int32(stage)))
        cp.cuda.Stream.null.synchronize()

    def _intt(self, d_poly):
        """Inverse NTT in-place on GPU."""
        import math
        log_n = int(math.log2(N))
        for stage in range(log_n):
            self._ntt_inv(
                _grid(N // 2), (BLOCK,),
                (d_poly, self.d_inv_roots, np.int32(N), np.int32(stage)))
        self._scale(_grid(N), (BLOCK,),
                    (d_poly, np.uint32(self.inv_n), np.int32(N)))
        cp.cuda.Stream.null.synchronize()

    def poly_mul_ntt(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NTT-based polynomial multiplication mod (X^N+1, Q)."""
        d_a = cp.asarray(a.copy())
        d_b = cp.asarray(b.copy())
        self._ntt(d_a)
        self._ntt(d_b)
        d_c = cp.zeros(N, dtype=cp.uint32)
        self._pointwise(_grid(N), (BLOCK,),
                        (d_a, d_b, d_c, np.int32(N)))
        self._intt(d_c)
        return cp.asnumpy(d_c)

    # ── Encrypt / Decrypt ─────────────────────────────────────────────────────

    def encrypt(self, message: np.ndarray) -> tuple:
        assert message.max() < T
        msg_gpu = cp.asarray(message.astype(np.uint32))
        err_gpu = cp.asarray(np.random.randint(0, 4, N, dtype=np.uint32))
        ct0 = cp.zeros(N, dtype=cp.uint32)
        ct1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._encrypt(_grid(N), (BLOCK,),
                      (msg_gpu, err_gpu, ct0, ct1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] Encrypted in {ms:.3f}ms")
        return ct0, ct1

    def decrypt(self, ct0, ct1) -> np.ndarray:
        out = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._decrypt(_grid(N), (BLOCK,), (ct0, out, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] Decrypted in {ms:.3f}ms")
        return cp.asnumpy(out)

    # ── Homomorphic operations ────────────────────────────────────────────────

    def he_add(self, ct_a, ct_b) -> tuple:
        """Add two ciphertexts without decrypting."""
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._he_add(_grid(N), (BLOCK,),
                     (ct_a[0], ct_a[1], ct_b[0], ct_b[1],
                      out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] HE ADD in {ms:.3f}ms")
        return out0, out1

    def he_mul_ct(self, ct_a, ct_b) -> tuple:
        """
        Full ciphertext-ciphertext multiplication with relinearization.
        ct_a * ct_b -> degree-2 result -> relinearize back to degree 1.
        """
        t0 = time.perf_counter()

        a0, a1 = cp.asnumpy(ct_a[0]), cp.asnumpy(ct_a[1])
        b0, b1 = cp.asnumpy(ct_b[0]), cp.asnumpy(ct_b[1])

        # Degree-2 ciphertext components via NTT multiplication
        # c0 = a0*b0, c1 = a0*b1 + a1*b0, c2 = a1*b1
        c0 = self.poly_mul_ntt(a0, b0)
        c1_part1 = self.poly_mul_ntt(a0, b1)
        c1_part2 = self.poly_mul_ntt(a1, b0)
        c2 = self.poly_mul_ntt(a1, b1)

        # c1 = c1_part1 + c1_part2 mod Q
        c1 = (c1_part1.astype(np.int64) + c1_part2.astype(np.int64)) % Q
        c1 = c1.astype(np.uint32)

        # Scale down by delta to keep in correct noise range
        c0 = np.array([(x * T + Q//2) // Q % Q for x in c0], dtype=np.uint32)
        c1 = np.array([(x * T + Q//2) // Q % Q for x in c1], dtype=np.uint32)
        c2_scaled = np.array([(x * T + Q//2) // Q % Q for x in c2], dtype=np.uint32)

        # Relinearization: absorb c2 using relin keys
        d_c2  = cp.asarray(c2_scaled)
        relin0 = cp.zeros(N, dtype=cp.uint32)
        relin1 = cp.zeros(N, dtype=cp.uint32)
        self._relin(_grid(N), (BLOCK,),
                    (d_c2, self.d_rlk0, self.d_rlk1,
                     relin0, relin1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()

        # Final ct = (c0 + relin0, c1 + relin1) mod Q
        d_c0 = cp.asarray(c0)
        d_c1 = cp.asarray(c1)
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        self._he_add(_grid(N), (BLOCK,),
                     (d_c0, d_c1, relin0, relin1,
                      out0, out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()

        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] HE MUL (ct*ct) + Relinearization in {ms:.3f}ms")
        return out0, out1

    def modswitch_down(self, ct) -> tuple:
        """Switch ciphertext to smaller modulus Q' to reduce noise."""
        out0 = cp.zeros(N, dtype=cp.uint32)
        out1 = cp.zeros(N, dtype=cp.uint32)
        t0 = time.perf_counter()
        self._modswitch_dn(_grid(N), (BLOCK,), (ct[0], out0, np.int32(N)))
        self._modswitch_dn(_grid(N), (BLOCK,), (ct[1], out1, np.int32(N)))
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] Modulus switch Q={Q} -> Q'={Q_PRIME} in {ms:.3f}ms")
        return out0, out1

    def benchmark(self, n_ops=1000):
        print(f"\n[cuFHE] Benchmarking {n_ops} HE additions...")
        msg = np.random.randint(0, T, N, dtype=np.uint32)
        ct_a = self.encrypt(msg)
        ct_b = self.encrypt(msg)
        t0 = time.perf_counter()
        for _ in range(n_ops):
            self.he_add(ct_a, ct_b)
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] {n_ops} HE ADD: {ms:.1f}ms ({n_ops/ms*1000:.0f} ops/sec)")

        print(f"\n[cuFHE] Benchmarking 100 HE multiplications (ct*ct)...")
        t0 = time.perf_counter()
        for _ in range(100):
            self.he_mul_ct(ct_a, ct_b)
        ms = (time.perf_counter()-t0)*1000
        print(f"[cuFHE] 100 HE MUL: {ms:.1f}ms ({100/ms*1000:.0f} ops/sec)")
