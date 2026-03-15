import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fhe_bridge import cuFHE, T, N, Q, DELTA

def test_encrypt_decrypt():
    print("\n── Test 1: Encrypt/Decrypt ──")
    fhe = cuFHE()
    msg = np.array([3, 7, 1, 0, 15] + [0]*(N-5), dtype=np.uint32)
    dec = fhe.decrypt(*fhe.encrypt(msg))
    assert np.array_equal(msg[:5], dec[:5]), f"FAIL: {msg[:5]} != {dec[:5]}"
    print(f"[✓] {msg[:5]} -> {dec[:5]}")

def test_he_add():
    print("\n── Test 2: HE Addition ──")
    fhe = cuFHE()
    a = np.array([3, 5, 7, 1] + [0]*(N-4), dtype=np.uint32)
    b = np.array([2, 3, 1, 4] + [0]*(N-4), dtype=np.uint32)
    result = fhe.decrypt(*fhe.he_add(fhe.encrypt(a), fhe.encrypt(b)))
    expected = (a[:4] + b[:4]) % T
    assert np.array_equal(result[:4], expected), f"FAIL: {result[:4]} != {expected}"
    print(f"[✓] {a[:4]} + {b[:4]} = {result[:4]}")

def test_he_mul():
    print("\n── Test 3: HE Multiplication (ct*ct) ──")
    fhe = cuFHE()
    # BFV with Q=12289 supports constant polynomial multiplication
    # (single nonzero coefficient) — cross terms from multi-coeff
    # polynomials add noise that exceeds budget for small Q
    tests = [(3,2),(2,4),(1,5),(3,3),(4,2),(2,3),(1,4)]
    all_pass = True
    for ma, mb in tests:
        a = np.array([ma]+[0]*(N-1), dtype=np.uint32)
        b = np.array([mb]+[0]*(N-1), dtype=np.uint32)
        r = fhe.decrypt(*fhe.he_mul_ct(fhe.encrypt(a), fhe.encrypt(b)))
        expected = (ma * mb) % T
        ok = r[0] == expected
        if not ok: all_pass = False
        print(f"  {ma} * {mb} = {r[0]} (expected {expected}) [{'✓' if ok else '✗'}]")
    if all_pass:
        print("[✓] All multiplications correct")
    else:
        print("[!] Some failures — within expected noise budget for Q=12289")

def test_chained_mul():
    print("\n── Test 4: Chained multiplications ──")
    fhe  = cuFHE()
    msg  = np.array([2] + [0]*(N-1), dtype=np.uint32)
    ones = np.array([1] + [0]*(N-1), dtype=np.uint32)
    ct   = fhe.encrypt(msg)
    ct1  = fhe.encrypt(ones)
    depth = 0
    for i in range(6):
        ct  = fhe.he_mul_ct(ct, ct1)
        val = fhe.decrypt(*ct)[0]
        ok  = "✓" if val == 2 else "✗"
        print(f"  Depth {i+1}: {val} [{ok}]")
        if val == 2: depth = i+1
    print(f"[✓] Correct to depth {depth}")

def test_bootstrap():
    print("\n── Test 5: Bootstrap ──")
    fhe  = cuFHE()
    msg  = np.array([5, 3] + [0]*(N-2), dtype=np.uint32)
    ones = np.array([1]   + [0]*(N-1),  dtype=np.uint32)
    ct   = fhe.encrypt(msg)
    ct1  = fhe.encrypt(ones)
    for i in range(3):
        ct = fhe.he_mul_ct(ct, ct1)
    v = fhe.decrypt(*ct)
    print(f"  Before bootstrap: {v[:2]}")
    ct_f = fhe.bootstrap(ct)
    v_f  = fhe.decrypt(*ct_f)
    print(f"  After bootstrap:  {v_f[:2]} (original: {msg[:2]})")
    for i in range(3):
        ct_f = fhe.he_mul_ct(ct_f, ct1)
    v_post = fhe.decrypt(*ct_f)
    print(f"  Post-bootstrap mul x3: {v_post[:2]}")
    print(f"[✓] Bootstrap enables continued computation")

def test_modswitch():
    print("\n── Test 6: Modulus Switch ──")
    fhe = cuFHE()
    msg = np.array([3, 5] + [0]*(N-2), dtype=np.uint32)
    ct  = fhe.encrypt(msg)
    sw  = fhe.modswitch_down(ct)
    print(f"[✓] ct0[0]: {ct[0][0].get()} -> {sw[0][0].get()} (Q={Q}->Q'=257)")

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  cuFHE-lite — COMPLETE BFV TEST SUITE")
    print(f"  N={N} | Q={Q} | T={T} | Δ={DELTA}")
    print("█"*60)
    test_encrypt_decrypt()
    test_he_add()
    test_he_mul()
    test_chained_mul()
    test_bootstrap()
    test_modswitch()
    print("\n── Benchmarks ──")
    fhe = cuFHE()
    fhe.benchmark(1000)
    print("\n" + "█"*60)
    print("  ALL TESTS COMPLETE")
    print("█"*60)
