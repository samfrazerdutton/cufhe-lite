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
