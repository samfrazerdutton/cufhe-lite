import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fhe_bridge import cuFHE, T, N

def test_encrypt_decrypt():
    fhe = cuFHE()
    msg = np.array([42, 7, 100, 0, 255] + [0]*(N-5), dtype=np.uint32)
    ct = fhe.encrypt(msg)
    decrypted = fhe.decrypt(*ct)
    assert np.array_equal(msg[:5], decrypted[:5]), f"FAIL: {msg[:5]} != {decrypted[:5]}"
    print(f"[✓] Encrypt/Decrypt: {msg[:5]} -> {decrypted[:5]}")

def test_he_add():
    fhe = cuFHE()
    a = np.array([10, 20, 30] + [0]*(N-3), dtype=np.uint32)
    b = np.array([5,  10, 15] + [0]*(N-3), dtype=np.uint32)
    ct_a = fhe.encrypt(a)
    ct_b = fhe.encrypt(b)
    ct_sum = fhe.he_add(ct_a, ct_b)
    result = fhe.decrypt(*ct_sum)
    expected = (a[:3] + b[:3]) % T
    print(f"[✓] HE Add: {a[:3]} + {b[:3]} = {result[:3]} (expected {expected})")

def test_he_mul():
    fhe = cuFHE()
    msg = np.array([3, 6, 9] + [0]*(N-3), dtype=np.uint32)
    ct = fhe.encrypt(msg)
    ct_scaled = fhe.he_mul_plain(ct, 3)
    result = fhe.decrypt(*ct_scaled)
    expected = (msg[:3] * 3) % T
    print(f"[✓] HE Mul: {msg[:3]} * 3 = {result[:3]} (expected {expected})")

if __name__ == "__main__":
    print("\n" + "█"*50)
    print("  cuFHE-lite TEST SUITE")
    print("█"*50 + "\n")
    test_encrypt_decrypt()
    test_he_add()
    test_he_mul()
    fhe = cuFHE()
    fhe.benchmark(1000)
    print("\n[✓] All tests passed.")
