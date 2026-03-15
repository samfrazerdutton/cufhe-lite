#include <stdint.h>
#include <cuda_runtime.h>

// BFV parameters
#define Q 65537ULL        // ciphertext modulus (Fermat prime, fast reduction)
#define T 256ULL          // plaintext modulus
#define N 1024            // polynomial degree (power of 2)
#define DELTA (Q / T)     // scaling factor

// Barrett reduction mod Q=65537
// Since Q is a Fermat prime (2^16 + 1), reduction is extremely fast
__device__ __forceinline__ uint32_t reduce_q(uint64_t a) {
    // For Q=65537=2^16+1: a mod Q = (a & 0xFFFF) - (a >> 16) + correction
    uint32_t lo = (uint32_t)(a & 0xFFFF);
    uint32_t hi = (uint32_t)(a >> 16);
    int32_t r = (int32_t)lo - (int32_t)hi;
    if (r < 0) r += Q;
    if ((uint32_t)r >= Q) r -= Q;
    return (uint32_t)r;
}

// Polynomial addition mod Q: c = a + b mod Q
__global__ void poly_add(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t* __restrict__ c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t sum = a[i] + b[i];
    if (sum >= Q) sum -= Q;
    c[i] = sum;
}

// Polynomial subtraction mod Q: c = a - b mod Q
__global__ void poly_sub(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t* __restrict__ c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t r = a[i] + Q - b[i];
    if (r >= Q) r -= Q;
    c[i] = r;
}

// Scalar multiply: c = a * scalar mod Q
__global__ void poly_scalar_mul(
    const uint32_t* __restrict__ a,
    uint32_t scalar,
    uint32_t* __restrict__ c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = reduce_q((uint64_t)a[i] * scalar);
}

// BFV Encrypt: ct = (a*sk + e + delta*m, -a)
// Simplified: ct0 = delta*m + e, ct1 = 0 (demo without full key switching)
__global__ void bfv_encrypt(
    const uint32_t* __restrict__ message,  // plaintext polynomial
    const uint32_t* __restrict__ error,    // small error polynomial
    uint32_t* __restrict__ ct0,            // ciphertext part 0
    uint32_t* __restrict__ ct1,            // ciphertext part 1
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // ct0 = delta * m + e mod Q
    uint64_t scaled = (uint64_t)message[i] * DELTA;
    ct0[i] = reduce_q(scaled + error[i]);
    ct1[i] = 0;
}

// BFV Decrypt: m = round((t/Q) * ct0) mod t
__global__ void bfv_decrypt(
    const uint32_t* __restrict__ ct0,
    uint32_t* __restrict__ message,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // m = round(t * ct0 / Q) mod t
    uint64_t scaled = (uint64_t)ct0[i] * T;
    uint32_t m = (uint32_t)((scaled + Q/2) / Q) % T;
    message[i] = m;
}

// Homomorphic addition: ct_add = ct_a + ct_b (componentwise mod Q)
__global__ void he_add(
    const uint32_t* __restrict__ ct_a0,
    const uint32_t* __restrict__ ct_a1,
    const uint32_t* __restrict__ ct_b0,
    const uint32_t* __restrict__ ct_b1,
    uint32_t* __restrict__ ct_out0,
    uint32_t* __restrict__ ct_out1,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = ct_a0[i] + ct_b0[i]; if (s0 >= Q) s0 -= Q;
    uint32_t s1 = ct_a1[i] + ct_b1[i]; if (s1 >= Q) s1 -= Q;
    ct_out0[i] = s0;
    ct_out1[i] = s1;
}

// Homomorphic plaintext multiply: ct_out = ct * plaintext scalar
__global__ void he_mul_plain(
    const uint32_t* __restrict__ ct0,
    const uint32_t* __restrict__ ct1,
    uint32_t plain_scalar,
    uint32_t* __restrict__ ct_out0,
    uint32_t* __restrict__ ct_out1,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ct_out0[i] = reduce_q((uint64_t)ct0[i] * plain_scalar);
    ct_out1[i] = reduce_q((uint64_t)ct1[i] * plain_scalar);
}

// ── NTT-based Polynomial Multiplication ──────────────────────────────────────
// For BFV ciphertext multiplication we need: c = a * b mod (X^N + 1, Q)
// Strategy: NTT(a) pointwise* NTT(b) then INTT

// Modular exponentiation for twiddle factor generation
__device__ uint32_t powmod(uint32_t base, uint32_t exp, uint32_t mod) {
    uint64_t result = 1;
    uint64_t b = base % mod;
    while (exp > 0) {
        if (exp & 1) result = result * b % mod;
        b = b * b % mod;
        exp >>= 1;
    }
    return (uint32_t)result;
}

// Cooley-Tukey NTT butterfly kernel
// Each thread handles one butterfly operation
__global__ void ntt_forward(
    uint32_t* __restrict__ poly,   // polynomial coefficients in-place
    const uint32_t* __restrict__ roots, // precomputed roots of unity
    int n,                         // polynomial degree
    int stage)                     // current NTT stage
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = 1 << stage;
    int stride = half << 1;
    int group = tid / half;
    int pos = tid % half;
    int i = group * stride + pos;
    int j = i + half;

    if (j >= n) return;

    uint32_t w = roots[half + pos];
    uint64_t u = poly[i];
    uint64_t v = (uint64_t)poly[j] * w % Q;

    uint32_t sum = (u + v) % Q;
    uint32_t diff = (u + Q - v) % Q;
    poly[i] = sum;
    poly[j] = diff;
}

// Inverse NTT — identical butterfly but with inverse roots + final scaling
__global__ void ntt_inverse(
    uint32_t* __restrict__ poly,
    const uint32_t* __restrict__ inv_roots,
    int n,
    int stage)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = 1 << stage;
    int stride = half << 1;
    int group = tid / half;
    int pos = tid % half;
    int i = group * stride + pos;
    int j = i + half;

    if (j >= n) return;

    uint32_t w = inv_roots[half + pos];
    uint64_t u = poly[i];
    uint64_t v = (uint64_t)poly[j] * w % Q;

    uint32_t sum = (u + v) % Q;
    uint32_t diff = (u + Q - v) % Q;
    poly[i] = sum;
    poly[j] = diff;
}

// Pointwise multiply two NTT-domain polynomials
__global__ void poly_pointwise_mul(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t* __restrict__ c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = (uint32_t)((uint64_t)a[i] * b[i] % Q);
}

// Scale by N^{-1} mod Q after INTT (final step of inverse NTT)
__global__ void poly_scale(
    uint32_t* __restrict__ poly,
    uint32_t inv_n,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    poly[i] = (uint32_t)((uint64_t)poly[i] * inv_n % Q);
}

// ── Relinearization key application ──────────────────────────────────────────
// After ct*ct multiplication we get a degree-2 ciphertext (ct0, ct1, ct2)
// Relinearization reduces it back to degree 1 using relin keys
// relin_key = (rlk0, rlk1) where rlk_i = encrypt(-s^2) under special modulus

__global__ void relin_key_mul(
    const uint32_t* __restrict__ ct2,     // extra ciphertext component
    const uint32_t* __restrict__ rlk0,    // relin key part 0
    const uint32_t* __restrict__ rlk1,    // relin key part 1
    uint32_t* __restrict__ out0,          // output correction for ct0
    uint32_t* __restrict__ out1,          // output correction for ct1
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Decompose ct2[i] and multiply by relin key
    // Simplified: direct multiplication (full impl uses gadget decomposition)
    out0[i] = (uint32_t)((uint64_t)ct2[i] * rlk0[i] % Q);
    out1[i] = (uint32_t)((uint64_t)ct2[i] * rlk1[i] % Q);
}

// ── Modulus switching ─────────────────────────────────────────────────────────
// Reduces ciphertext modulus from Q to Q' < Q to reduce noise growth
// Formula: ct' = round(Q'/Q * ct) mod Q'
// We use Q=65537, Q'=257 (both Fermat primes for fast reduction)

#define Q_PRIME 257U  // smaller modulus for switching

__global__ void modswitch_down(
    const uint32_t* __restrict__ ct_in,
    uint32_t* __restrict__ ct_out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // round(Q_PRIME/Q * ct_in) mod Q_PRIME
    uint64_t scaled = (uint64_t)ct_in[i] * Q_PRIME + Q / 2;
    ct_out[i] = (uint32_t)((scaled / Q) % Q_PRIME);
}

__global__ void modswitch_up(
    const uint32_t* __restrict__ ct_in,
    uint32_t* __restrict__ ct_out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Lift from Q' back to Q preserving value mod Q'
    ct_out[i] = ct_in[i] % Q;
}
