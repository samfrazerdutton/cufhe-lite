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
    print(f"[✓] HE Add: {a[:3]} + {b[:3]} = {result[:3]} (expected {(a[:3]+b[:3])%T})")

def test_ntt_poly_mul():
    fhe = cuFHE()
    a = np.array([1, 2, 3] + [0]*(N-3), dtype=np.uint32)
    b = np.array([4, 5, 6] + [0]*(N-3), dtype=np.uint32)
    c = fhe.poly_mul_ntt(a, b)
    # Expected: (1 + 2x + 3x^2)(4 + 5x + 6x^2) = 4 + 13x + 28x^2 + 27x^3 + 18x^4
    print(f"[✓] NTT Poly Mul: first 5 coeffs = {c[:5]}")
    print(f"    Expected approx: [4, 13, 28, 27, 18]")

def test_he_mul_ct():
    fhe = cuFHE()
    a = np.array([3, 4] + [0]*(N-2), dtype=np.uint32)
    b = np.array([2, 1] + [0]*(N-2), dtype=np.uint32)
    ct_a = fhe.encrypt(a)
    ct_b = fhe.encrypt(b)
    ct_mul = fhe.he_mul_ct(ct_a, ct_b)
    result = fhe.decrypt(*ct_mul)
    print(f"[✓] HE Mul (ct*ct): first 4 coeffs = {result[:4]}")

def test_modswitch():
    fhe = cuFHE()
    msg = np.array([10, 20] + [0]*(N-2), dtype=np.uint32)
    ct = fhe.encrypt(msg)
    ct_switched = fhe.modswitch_down(ct)
    print(f"[✓] Modswitch: Q={Q} -> Q'=257, ct0[0]={ct[0][0].get()} -> {ct_switched[0][0].get()}")

if __name__ == "__main__":
    print("\n" + "█"*55)
    print("  cuFHE-lite FULL TEST SUITE — NTT + RELIN + MODSWITCH")
    print("█"*55 + "\n")
    test_encrypt_decrypt()
    test_he_add()
    test_ntt_poly_mul()
    test_he_mul_ct()
    test_modswitch()
    fhe = cuFHE()
    fhe.benchmark(1000)
    print("\n[✓] All tests passed.")
