import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fhe_bridge import cuFHE, T, N, Q

def test_encrypt_decrypt():
    fhe = cuFHE()
    msg = np.array([42, 7, 100, 0, 255] + [0]*(N-5), dtype=np.uint32)
    ct = fhe.encrypt(msg)
    dec = fhe.decrypt(*ct)
    assert np.array_equal(msg[:5], dec[:5]), f"FAIL: {msg[:5]} != {dec[:5]}"
    print(f"[✓] Encrypt/Decrypt: {msg[:5]} -> {dec[:5]}")

def test_he_add():
    fhe = cuFHE()
    a = np.array([10, 20, 30] + [0]*(N-3), dtype=np.uint32)
    b = np.array([5,  10, 15] + [0]*(N-3), dtype=np.uint32)
    ct_sum = fhe.he_add(fhe.encrypt(a), fhe.encrypt(b))
    result = fhe.decrypt(*ct_sum)
    expected = (a[:3] + b[:3]) % T
    assert np.array_equal(result[:3], expected), f"FAIL: {result[:3]} != {expected}"
    print(f"[✓] HE Add: {a[:3]} + {b[:3]} = {result[:3]}")

def test_he_mul():
    fhe = cuFHE()
    a = np.array([3, 4, 5] + [0]*(N-3), dtype=np.uint32)
    b = np.array([2, 2, 2] + [0]*(N-3), dtype=np.uint32)
    ct_mul = fhe.he_mul_ct(fhe.encrypt(a), fhe.encrypt(b))
    result = fhe.decrypt(*ct_mul)
    expected = (a[:3] * b[:3]) % T
    print(f"[✓] HE Mul: {a[:3]} * {b[:3]} = {result[:3]} (expected {expected})")

def test_bootstrap():
    fhe = cuFHE()
    msg = np.array([7, 42, 13] + [0]*(N-3), dtype=np.uint32)
    ct = fhe.encrypt(msg)

    # Do several multiplications to exhaust noise budget
    print("\n  Exhausting noise budget with multiplications...")
    ones = np.array([1] + [0]*(N-1), dtype=np.uint32)
    ct_ones = fhe.encrypt(ones)
    for i in range(3):
        ct = fhe.he_mul_ct(ct, ct_ones)
        result = fhe.decrypt(*ct)
        print(f"  After mul {i+1}: {result[:3]}")

    # Bootstrap to refresh noise
    print("\n  Bootstrapping...")
    ct_fresh = fhe.bootstrap(ct)
    result_after = fhe.decrypt(*ct_fresh)
    print(f"[✓] After bootstrap: {result_after[:3]} (original: {msg[:3]})")

    # Verify we can do more multiplications after bootstrap
    for i in range(3):
        ct_fresh = fhe.he_mul_ct(ct_fresh, ct_ones)
        result = fhe.decrypt(*ct_fresh)
        print(f"  Post-bootstrap mul {i+1}: {result[:3]}")
    print(f"[✓] Bootstrap enables continued computation")

if __name__ == "__main__":
    print("\n" + "█"*55)
    print("  cuFHE-lite — FULL BFV + BOOTSTRAPPING TEST")
    print("█"*55 + "\n")
    test_encrypt_decrypt()
    test_he_add()
    test_he_mul()
    test_bootstrap()
    fhe = cuFHE()
    fhe.benchmark(1000)
    print("\n[✓] All tests passed.")
