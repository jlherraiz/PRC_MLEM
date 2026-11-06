// prc.cu -- GPU positron-range forward/adjoint projection and image correction.
// Density-integrated straight rays with per-source normalization.
// Richardson-Lucy with an RDP prior and optional Biggs-Andrews acceleration.
// Prior/accelerated updates are empirical; no global convergence guarantee.
// Full-data iterations, without ordered subsets.
// Build: bash build.sh. Run ./prc without arguments for usage.

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                                 \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

// ------------------------------------------------------------------------------------ isotopes
// Survival S(r) = sum_b w_b exp(-p_b(r)), p_b(r) = (a r)^(1+f) + (b r)^2.5 + (c r)^3.5: the fraction of
// positrons that travel farther than the water-equivalent distance r [mm] (Paneque-Yunta et al.,
// Phys. Med. Biol. 70 (2025) 165013, Tables 4 and 5).
struct Branch {
    float w, a, b, c, f;
};
struct Isotope {
    const char* name;
    int n;
    Branch br[3];
};

static const Isotope kIsotopes[] = {
    {"11C", 1, {{1.000f, 0.777f, 0.468f, 0.217f, 0.010f}}},
    {"13N", 1, {{1.000f, 0.537f, 0.347f, 0.193f, 0.050f}}},
    {"15O", 1, {{1.000f, 0.295f, 0.221f, 0.133f, 0.118f}}},
    {"18F", 1, {{1.000f, 1.612f, 0.703f, 0.526f, 0.011f}}},
    {"64Cu", 1, {{1.000f, 1.494f, 0.000f, 0.631f, 0.251f}}},
    {"68Ga", 2, {{0.014f, 1.048f, 0.000f, 0.475f, 0.271f}, {0.986f, 0.289f, 0.137f, 0.159f, 0.322f}}},
    {"82Rb", 2, {{0.136f, 0.191f, 0.084f, 0.114f, 0.408f}, {0.857f, 0.135f, 0.063f, 0.087f, 0.451f}}},
    {"124I", 3, {{0.013f, 1.007f, 0.000f, 0.421f, 0.443f}, {0.515f, 0.412f, 0.000f, 0.210f, 0.458f},
                 {0.471f, 0.241f, 0.000f, 0.150f, 0.369f}}},
};

__constant__ Branch c_branch[3];
__constant__ int c_nbranch;
constexpr int kRangeBins = 4096;      // 16 KiB table of S on [0, cutoff]: stays in L1
__constant__ const float* c_survival;  // the table; nullptr = evaluate S analytically
__constant__ float c_survival_scale;   // bins per mm

__host__ __device__ inline float survival(const Branch* br, int n, float r) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float x = br[i].b * r, y = br[i].c * r;  // x^2.5 = x^2 sqrt(x), y^3.5 = y^3 sqrt(y)
        s += br[i].w * expf(-(powf(br[i].a * r, 1.0f + br[i].f) + x * x * sqrtf(x) + y * y * y * sqrtf(y)));
    }
    return s;
}

__device__ float survival_at(float w) {  // 0 <= w <= cutoff
    if (!c_survival) return survival(c_branch, c_nbranch, w);
    const float u = w * c_survival_scale;
    const int i = min((int)u, kRangeBins - 1);
    const float lo = __ldg(c_survival + i), hi = __ldg(c_survival + i + 1);
    return fmaf(fminf(u - i, 1.0f), hi - lo, lo);
}

// Water-equivalent distance that a fraction `fraction` of the positrons does not exceed.
static float range_cutoff(const Isotope& iso, float fraction) {
    float lo = 0.0f, hi = 1.0f;
    while (1.0f - survival(iso.br, iso.n, hi) < fraction && hi < 256.0f) hi *= 2.0f;
    for (int it = 0; it < 64; ++it) {
        const float mid = 0.5f * (lo + hi);
        (1.0f - survival(iso.br, iso.n, mid) < fraction ? lo : hi) = mid;
    }
    return hi;
}

// ------------------------------------------------------------------------------------ geometry
struct Grid {
    int nx, ny, nz;
    float sx, sy, sz;   // voxel size [mm]
    float cutoff;       // water-equivalent cutoff [mm]
    float support_rho;  // voxels with density >= support_rho hold activity
    float air_cap;      // a ray is dropped after this many mm of consecutive air
};

constexpr float kMinDensity = 1.0e-3f;
constexpr int kNearR = 2, kNearSide = 2 * kNearR + 1, kNear = kNearSide * kNearSide * kNearSide;  // 5x5x5
constexpr int kNearRays = 2048;  // near-block error < 1% of the blur at 2-4 mm (exact water kernel)
constexpr int kThreads = 128;    // projector block: 4 warps, one source voxel each

__device__ inline float density_at(cudaTextureObject_t tex, int i, int j, int k) {
    return tex3D<float>(tex, i + 0.5f, j + 0.5f, k + 0.5f);
}

__device__ inline bool inside(const Grid& g, int i, int j, int k) {
    return i >= 0 && i < g.nx && j >= 0 && j < g.ny && k >= 0 && k < g.nz;
}

// Ray k of n. Rays 2m and 2m+1 start at +/- the m-th point of the R3 low-discrepancy sequence inside
// the source voxel (in voxels from its centre) and leave along +/- the m-th point of a Fibonacci lattice
// on the upper hemisphere. The mirrored pairs keep the kernel of a uniform medium centred on its source.
// Unsigned products wrap modulo 2^32, which gives the fractional parts of m * constant exactly.
__device__ inline void ray(int k, int n, float3* start, float3* dir) {
    const unsigned m = k >> 1;
    const float sign = (k & 1) ? -1.0f : 1.0f, z = 1.0f - (m + 0.5f) / ((n + 1) >> 1);
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z)), to_unit = 2.3283064e-10f;  // 2^-32
    float sn, cs;
    sincospif(2.0f * to_unit * (m * 2654435769u), &sn, &cs);  // 2^32 / golden ratio
    *dir = make_float3(sign * r * cs, sign * r * sn, sign * z);
    // R3: 1/2 + m / phi^d (mod 1) for d = 1, 2, 3, with phi^4 = phi + 1
    *start = make_float3(sign * (to_unit * (0x80000000u + m * 3518319155u) - 0.5f),
                         sign * (to_unit * (0x80000000u + m * 2882110345u) - 0.5f),
                         sign * (to_unit * (0x80000000u + m * 2360945575u) - 0.5f));
}

// Walks the ray from v + x (x in voxels from the centre of source voxel v) along the unit vector d,
// voxel by voxel (Amanatides-Woo). In each crossed voxel the water-equivalent path grows by rho times
// the length, and the positron annihilates there with probability S(w_in) - S(w_out).
// visit(i, j, k, p) receives the tissue voxels inside the near block (near = true; the walk ends at the
// block edge) or outside it (near = false). The walk also ends at the cutoff, at the volume edge or after
// air_cap mm of consecutive air.
template <class Visit>
__device__ void trace_ray(cudaTextureObject_t tex, const Grid& g, int3 v, float3 x, float3 d, bool near,
                          Visit&& visit) {
    const float big = 1.0e30f;
    const int di = d.x > 0.0f ? 1 : -1, dj = d.y > 0.0f ? 1 : -1, dk = d.z > 0.0f ? 1 : -1;
    // path length [mm] to the next x, y and z voxel faces, and between consecutive faces
    float tx = d.x != 0.0f ? (0.5f * di - x.x) * g.sx / d.x : big, dtx = d.x != 0.0f ? g.sx / fabsf(d.x) : big;
    float ty = d.y != 0.0f ? (0.5f * dj - x.y) * g.sy / d.y : big, dty = d.y != 0.0f ? g.sy / fabsf(d.y) : big;
    float tz = d.z != 0.0f ? (0.5f * dk - x.z) * g.sz / d.z : big, dtz = d.z != 0.0f ? g.sz / fabsf(d.z) : big;
    int i = v.x, j = v.y, k = v.z;
    float t = 0.0f, w = 0.0f, surv = 1.0f, air = 0.0f;
    for (;;) {
        const bool in_block = abs(i - v.x) <= kNearR && abs(j - v.y) <= kNearR && abs(k - v.z) <= kNearR;
        if (near && !in_block) return;
        const float t_next = fminf(tx, fminf(ty, tz));
        const float rho = density_at(tex, i, j, k);
        const float w_next = fminf(w + rho * (t_next - t), g.cutoff);
        const float surv_next = survival_at(w_next);
        if (rho >= g.support_rho) {
            air = 0.0f;
            if (in_block == near && surv > surv_next) visit(i, j, k, surv - surv_next);
        } else if ((air += t_next - t) > g.air_cap) {
            return;
        }
        if (w_next >= g.cutoff) return;
        t = t_next, w = w_next, surv = surv_next;
        if (t_next == tx) i += di, tx += dtx;
        else if (t_next == ty) j += dj, ty += dty;
        else k += dk, tz += dtz;
        if (!inside(g, i, j, k)) return;
    }
}

// ----------------------------------------------------------------------------- projector kernel
// One warp per tissue source voxel. The warp visits every (target t, weight w_st) of the source's
// kernel and hands them to Op:
//   NormOp    W_s = sum_t w_st
//   ForwardOp out_t += f_s w_st / W_s          (scatter with atomics)
//   AdjointOp out_s  = sum_t w_st in_t / W_s   (gather)
// The lanes split the near-block rays, adding their deposits per block voxel in shared memory (or read
// these weights from the cache), and then split the far-field rays.
struct ProjectorArgs {
    cudaTextureObject_t tex;
    Grid g;
    const int* active;  // tissue voxel indices
    int n_active;
    int n_rays;         // far-field rays
    float* near_cache;  // optional: kNear block weights per active source
};

template <class Op, bool BuildCache = false>
__global__ void project_kernel(ProjectorArgs a, Op op) {
    extern __shared__ float near_sum[];  // kNear per warp, allocated only when tracing the near block
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int src = blockIdx.x * (kThreads / 32) + warp;
    if (src >= a.n_active) return;
    const int s = a.active[src];
    if (!op.begin(src, s, lane)) return;  // decision is identical for the whole warp

    const Grid& g = a.g;
    const int3 v = make_int3(s % g.nx, (s / g.nx) % g.ny, s / (g.nx * g.ny));
    const bool trace_near = BuildCache || !a.near_cache;
    float* sum = near_sum + warp * kNear;
    float acc = 0.0f;

    if (trace_near) {
        for (int q = lane; q < kNear; q += 32) sum[q] = 0.0f;
        __syncwarp();
        for (int r = lane; r < kNearRays; r += 32) {
            float3 x, d;
            ray(r, kNearRays, &x, &d);
            trace_ray(a.tex, g, v, x, d, true, [&](int i, int j, int k, float p) {
                atomicAdd(sum + ((k - v.z + kNearR) * kNearSide + j - v.y + kNearR) * kNearSide + i - v.x + kNearR, p);
            });
        }
        __syncwarp();
    }
    for (int q = lane; q < kNear; q += 32) {
        const int i = v.x + q % kNearSide - kNearR, j = v.y + q / kNearSide % kNearSide - kNearR,
                  k = v.z + q / (kNearSide * kNearSide) - kNearR;
        if (!inside(g, i, j, k)) continue;
        const size_t cache_idx = (size_t)src * kNear + q;
        float w;
        if (trace_near) {
            w = sum[q] * ((float)a.n_rays / kNearRays);  // on the scale of one far-field ray
            if (BuildCache && a.near_cache) a.near_cache[cache_idx] = w;
        } else {
            w = a.near_cache[cache_idx];
        }
        if (w > 0.0f) acc += op.visit((k * g.ny + j) * g.nx + i, w);
    }

    for (int r = lane; r < a.n_rays; r += 32) {
        float3 x, d;
        ray(r, a.n_rays, &x, &d);
        trace_ray(a.tex, g, v, x, d, false, [&](int i, int j, int k, float p) {
            acc += op.visit((k * g.ny + j) * g.nx + i, p);
        });
    }

    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) op.end(src, s, acc);
}

struct NormOp {
    float* W;
    __device__ bool begin(int, int, int) { return true; }
    __device__ float visit(int, float w) { return w; }
    __device__ void end(int src, int, float acc) { W[src] = acc; }
};

struct ForwardOp {
    const float* f;
    const float* W;
    float* out;
    float scale;
    __device__ bool begin(int src, int s, int) {
        const float fs = f[s], ws = W[src];
        scale = ws > 0.0f ? fs / ws : 0.0f;
        return scale > 0.0f;
    }
    __device__ float visit(int t, float w) {
        atomicAdd(out + t, scale * w);
        return 0.0f;
    }
    __device__ void end(int, int, float) {}
};

struct AdjointOp {
    const float* in;
    const float* W;
    float* out;
    __device__ bool begin(int src, int s, int lane) {
        if (W[src] > 0.0f) return true;
        if (lane == 0) out[s] = 0.0f;
        return false;
    }
    __device__ float visit(int t, float w) { return w * in[t]; }
    __device__ void end(int src, int s, float acc) { out[s] = acc / W[src]; }
};

// ------------------------------------------------------------------------------- image kernels
__global__ void ratio_kernel(const float* g, const float* est, const unsigned char* support, float* ratio,
                             size_t n, float eps) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) ratio[i] = support[i] ? g[i] / fmaxf(est[i], eps) : 0.0f;
}

// Relative-difference prior on the 26 tissue neighbours. Distance weights use physical
// spacing, normalized by the smallest spacing (unchanged for isotropic voxels).
// Evaluate the gradient and a diagonal curvature at f, then solve the positive scalar
// quadratic update. Read f and write next separately: neighbours must see the same iterate.
// This is an empirically stable, count-rescaled MAP-EM update, not a global convergence proof.
__global__ void update_kernel(const float* f, const float* back, const unsigned char* support,
                              float* next, Grid g, float beta, float gamma, float eps) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)g.nx * g.ny * g.nz) return;
    if (!support[idx]) { next[idx] = 0.0f; return; }
    const float fj = f[idx], f_em = fj * back[idx];
    if (beta == 0.0f) { next[idx] = fmaxf(f_em, 0.0f); return; }
    const int i = idx % g.nx, j = (idx / g.nx) % g.ny, k = idx / ((size_t)g.nx * g.ny);
    float grad = 0.0f, curv = 0.0f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (!(dx | dy | dz) || !inside(g, i + dx, j + dy, k + dz)) continue;
                const size_t nb = idx + ((long)dz * g.ny + dy) * g.nx + dx;
                if (!support[nb]) continue;
                const float mx = dx * g.sx, my = dy * g.sy, mz = dz * g.sz;
                const float w = fminf(g.sx, fminf(g.sy, g.sz)) * rsqrtf(mx * mx + my * my + mz * mz);
                const float fk = f[nb], d = fj - fk, ad = fabsf(d);
                const float D = fj + fk + gamma * ad + eps, invD = 1.0f / D;
                grad += w * d * (fj + 3.0f * fk + gamma * ad + 2.0f * eps) * invD * invD;
                const float half_quad = (2.0f * (fj + fk + eps) + gamma * ad) * invD * invD;
                const float hess = 2.0f * (2.0f * fk + eps) * (2.0f * fk + eps) * invD * invD * invD;
                curv += 2.0f * w * fmaxf(half_quad, hess);
            }
    // Positive root of a*x^2 + B*x - f_em = 0. Use the stable form for each sign of B;
    // f_em == 0 can still have a positive solution when B < 0 (the prior fills a hole).
    const float a = beta * curv, B = 1.0f + beta * (grad - curv * fj);
    const float root = sqrtf(B * B + 4.0f * a * f_em);
    next[idx] = B >= 0.0f ? (f_em > 0.0f ? 2.0f * f_em / (B + root) : 0.0f)
                         : (root - B) / (2.0f * a);
}

__global__ void initialize_kernel(float* f, const unsigned char* support, size_t n, float floor) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n && support[i]) f[i] = fmaxf(f[i], floor);
}

// Biggs-Andrews prediction y = f + lambda (f - prev). It is kept above f / 2: a zero would stay
// zero under the multiplicative EM step.
__global__ void extrapolate_kernel(const float* f, const float* prev, float* y, float lambda, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(f[i] + lambda * (f[i] - prev[i]), 0.5f * f[i]);
}

// Normalize the new estimate and, when accelerating, turn the starting image into the step.
// Explicitly round the product before subtracting, matching separate scale/subtract kernels.
__global__ void finish_kernel(float* next, float* residual, float scale, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = __fmul_rn(next[i], scale);
        next[i] = value;
        if (residual) residual[i] = value - residual[i];
    }
}

// ------------------------------------------------------------------------------------ host side
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void read_raw(const std::string& path, std::vector<float>* v) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp || fread(v->data(), sizeof(float), v->size(), fp) != v->size() ||
        fgetc(fp) != EOF || ferror(fp)) {
        fprintf(stderr, "Expected exactly %zu float32 values in %s\n", v->size(), path.c_str());
        exit(EXIT_FAILURE);
    }
    fclose(fp);
}

static void write_raw(const std::string& path, const std::vector<float>& v) {
    for (float value : v) if (!std::isfinite(value)) {
        fprintf(stderr, "Non-finite output: %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp || fwrite(v.data(), sizeof(float), v.size(), fp) != v.size() || fclose(fp) != 0) {
        fprintf(stderr, "Cannot write %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
}

static cudaTextureObject_t make_density_texture(const std::vector<float>& rho, const Grid& g, cudaArray_t* arr) {
    const cudaExtent ext = make_cudaExtent(g.nx, g.ny, g.nz);
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMalloc3DArray(arr, &desc, ext));
    cudaMemcpy3DParms cp = {};
    cp.srcPtr = make_cudaPitchedPtr((void*)rho.data(), g.nx * sizeof(float), g.nx, g.ny);
    cp.dstArray = *arr;
    cp.extent = ext;
    cp.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&cp));
    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = *arr;
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;  // the density is constant inside each voxel
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res, &td, nullptr));
    return tex;
}

template <class T>
static T* device_alloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

static double device_sum(const float* d, size_t n) {
    const thrust::device_ptr<const float> p(d);
    return thrust::reduce(p, p + n, 0.0, thrust::plus<double>());
}

static double device_dot(const float* a, const float* b, size_t n) {
    const thrust::device_ptr<const float> pa(a), pb(b);
    return thrust::inner_product(pa, pa + n, pb, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
}

// Forward and adjoint operators with the per-source normalisation W computed once.
class Projector {
  public:
    Projector(const Grid& g, cudaTextureObject_t tex, const std::vector<int>& active, int n_rays,
              bool cache_near = false)
        : g_(g), tex_(tex), n_active_((int)active.size()), n_rays_(n_rays) {
        d_active_ = device_alloc<int>(n_active_);
        CUDA_CHECK(cudaMemcpy(d_active_, active.data(), n_active_ * sizeof(int), cudaMemcpyHostToDevice));
        d_W_ = device_alloc<float>(n_active_);
        // The near-block weights (kNear per source) are cached if they fit in a quarter of the free
        // memory; otherwise every pass traces them again.
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t bytes = (size_t)n_active_ * kNear * sizeof(float);
        if (cache_near && bytes <= free_bytes / 4 && cudaMalloc(&d_near_, bytes) != cudaSuccess) {
            d_near_ = nullptr;
            cudaGetLastError();  // optional cache: allocation failure falls back to recomputation
        }
        printf("near-field cache: %s (%.0f MiB)\n", d_near_ ? "on" : "off", bytes / 1048576.0);
        launch<true>(NormOp{d_W_});
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ~Projector() {
        cudaFree(d_active_);
        cudaFree(d_W_);
        cudaFree(d_near_);
    }

    // out = A f (out is overwritten)
    void forward(const float* f, float* out) const {
        CUDA_CHECK(cudaMemset(out, 0, (size_t)g_.nx * g_.ny * g_.nz * sizeof(float)));
        launch(ForwardOp{f, d_W_, out, 0.0f});
    }
    // out = A^T in on tissue voxels (other voxels are left untouched)
    void adjoint(const float* in, float* out) const { launch(AdjointOp{in, d_W_, out}); }

  private:
    template <bool BuildCache = false, class Op>
    void launch(Op op) const {
        const ProjectorArgs a{tex_, g_, d_active_, n_active_, n_rays_, d_near_};
        const int warps = kThreads / 32;
        // Shared memory only while tracing the near block: reserving it always would shrink the L1
        // cache that the far-field rays rely on (~10% slower passes).
        const size_t smem = BuildCache || !d_near_ ? warps * kNear * sizeof(float) : 0;
        project_kernel<Op, BuildCache><<<((size_t)n_active_ + warps - 1) / warps, kThreads, smem>>>(a, op);
        CUDA_CHECK(cudaGetLastError());
    }

    Grid g_;
    cudaTextureObject_t tex_;
    int n_active_, n_rays_;
    int* d_active_ = nullptr;
    float* d_W_ = nullptr;
    float* d_near_ = nullptr;
};

struct Options {
    std::string isotope, mode = "correct", input = "ACTIVITY.raw", density = "DENSITY.raw", output;
    int nx = 0, ny = 0, nz = 0;
    float dx = 0, dy = 0, dz = 0;
    int rays = 128, iterations = 20, save_every = 0;
    float beta = 0.2f, gamma = 20.0f, support_rho = 0.05f, air_cap_mm = 50.0f;
    bool reproject = false, cache_near = true, exact_range = false, accelerate = true;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s ISOTOPE NX NY NZ DX DY DZ [options]\n"
            "  --mode correct|forward|adjoint   (default correct)\n"
            "  --input FILE      blurred (correct) or source (forward/adjoint) image, default ACTIVITY.raw\n"
            "  --density FILE    density map in g/cm3, default DENSITY.raw\n"
            "  --output FILE     default ACTIVITY_PRC_V2.raw (correct), ACTIVITY_FWD.raw, ACTIVITY_ADJ.raw\n"
            "  --rays N          even far-field rays per voxel (default 128)\n"
            "  --iterations N    accelerated MAP-EM iterations (default 20)\n"
            "  --beta B          relative-difference prior strength (default 0.2; 0 = no prior)\n"
            "                    Add --no-accel for plain Richardson-Lucy.\n"
            "  --gamma G         RDP edge preservation, larger keeps sharper edges (default 20)\n"
            "  --support-rho R   minimum density of voxels that can hold activity (default 0.05)\n"
            "  --air-cap-mm L    a positron that crossed L mm of air has escaped (default 50)\n"
            "  --save-every K    also write the estimate every K iterations\n"
            "  --reproject       also write A f of the result (_REPROJ.raw)\n"
            "  --exact-range     evaluate the survival function analytically, not from the table (validation)\n"
            "  --no-cache-near   trace the near block again in every pass (less GPU memory, slower)\n"
            "  --no-accel        plain MAP-EM without Biggs-Andrews extrapolation (validation)\n"
            "Isotopes:",
            prog);
    for (const Isotope& iso : kIsotopes) fprintf(stderr, " %s", iso.name);
    fprintf(stderr, "\n");
}

static int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < INT_MIN || value > INT_MAX) {
        fprintf(stderr, "Invalid integer for %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static float parse_float(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const float value = strtof(text, &end);
    if (errno || end == text || *end || !std::isfinite(value)) {
        fprintf(stderr, "Invalid finite number for %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static Options parse(int argc, char** argv) {
    Options o;
    if (argc < 8) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    o.isotope = argv[1];
    o.nx = parse_int(argv[2], "NX"), o.ny = parse_int(argv[3], "NY"), o.nz = parse_int(argv[4], "NZ");
    o.dx = parse_float(argv[5], "DX"), o.dy = parse_float(argv[6], "DY"), o.dz = parse_float(argv[7], "DZ");
    for (int i = 8; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", key.c_str()); exit(EXIT_FAILURE); }
            return argv[++i];
        };
        if (key == "--mode") o.mode = value();
        else if (key == "--input") o.input = value();
        else if (key == "--density") o.density = value();
        else if (key == "--output") o.output = value();
        else if (key == "--rays") o.rays = parse_int(value(), key.c_str());
        else if (key == "--iterations") o.iterations = parse_int(value(), key.c_str());
        else if (key == "--beta") o.beta = parse_float(value(), key.c_str());
        else if (key == "--gamma") o.gamma = parse_float(value(), key.c_str());
        else if (key == "--support-rho") o.support_rho = parse_float(value(), key.c_str());
        else if (key == "--air-cap-mm") o.air_cap_mm = parse_float(value(), key.c_str());
        else if (key == "--save-every") o.save_every = parse_int(value(), key.c_str());
        else if (key == "--reproject") o.reproject = true;
        else if (key == "--no-cache-near") o.cache_near = false;
        else if (key == "--exact-range") o.exact_range = true;
        else if (key == "--no-accel") o.accelerate = false;
        else { fprintf(stderr, "Unknown option %s\n", key.c_str()); print_usage(argv[0]); exit(EXIT_FAILURE); }
    }
    if (o.nx <= 0 || o.ny <= 0 || o.nz <= 0 || o.dx <= 0 || o.dy <= 0 || o.dz <= 0 || o.rays < 2 || (o.rays & 1) ||
        o.rays > INT_MAX - 31 || o.iterations < 0 || o.iterations == INT_MAX || o.beta < 0 || o.gamma < 0 ||
        o.support_rho < 0 || o.air_cap_mm <= 0 || o.save_every < 0 ||
        (o.mode != "correct" && o.mode != "forward" && o.mode != "adjoint")) {
        fprintf(stderr, "Invalid geometry or solver parameters\n");
        exit(EXIT_FAILURE);
    }
    // Projector voxel indices are signed 32-bit; check products before multiplying them.
    if (o.nx > INT_MAX / o.ny || o.nx * o.ny > INT_MAX / o.nz) {
        fprintf(stderr, "Volume exceeds the INT_MAX voxel index limit\n");
        exit(EXIT_FAILURE);
    }
    if (o.output.empty())
        o.output = o.mode == "correct" ? "ACTIVITY_PRC_V2.raw" : o.mode == "forward" ? "ACTIVITY_FWD.raw" : "ACTIVITY_ADJ.raw";
    return o;
}

static std::string with_suffix(const std::string& path, const std::string& suffix) {
    const size_t dot = path.rfind('.');
    const size_t slash = path.find_last_of("/\\");
    const size_t start = slash == std::string::npos ? 0 : slash + 1;
    return dot == std::string::npos || dot <= start ? path + suffix : path.substr(0, dot) + suffix + path.substr(dot);
}

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    const Isotope* iso = nullptr;
    for (const Isotope& it : kIsotopes)
        if (o.isotope == it.name) iso = &it;
    if (!iso) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    Isotope model = *iso;  // the smallest branch absorbs the rounding of the published weights
    float wsum = 0.0f;
    int smallest = 0;
    for (int b = 0; b < model.n; ++b) {
        wsum += model.br[b].w;
        if (model.br[b].w < model.br[smallest].w) smallest = b;
    }
    model.br[smallest].w += 1.0f - wsum;

    Grid g{o.nx, o.ny, o.nz, o.dx, o.dy, o.dz, 0, o.support_rho, 0};
    g.cutoff = range_cutoff(model, 0.9995f);
    g.air_cap = o.air_cap_mm;
    const size_t n = (size_t)g.nx * g.ny * g.nz;

    const double t0 = now_s();
    std::vector<float> rho(n), img(n);
    read_raw(o.density, &rho);
    read_raw(o.input, &img);
    std::vector<int> active;
    std::vector<unsigned char> support(n);
    double input_sum = 0.0, support_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(rho[i]) || !std::isfinite(img[i])) {
            fprintf(stderr, "Non-finite density or activity at voxel %zu\n", i);
            return EXIT_FAILURE;
        }
        rho[i] = std::max(rho[i], kMinDensity);
        img[i] = std::max(img[i], 0.0f);
        support[i] = rho[i] >= o.support_rho;
        input_sum += img[i];
        if (support[i]) {
            active.push_back((int)i);
            support_sum += img[i];
        } else {
            img[i] = 0.0f;
        }
    }
    printf("isotope %s, grid %dx%dx%d, voxel %.3fx%.3fx%.3f mm, cutoff %.3f mm (water), air cap %.1f mm\n",
           o.isotope.c_str(), g.nx, g.ny, g.nz, g.sx, g.sy, g.sz, g.cutoff, g.air_cap);
    printf("tissue voxels %zu (%.1f%%), input counts in tissue %.6g of %.6g (%.2f%%)\n", active.size(),
           100.0 * active.size() / n, support_sum, input_sum, input_sum > 0 ? 100.0 * support_sum / input_sum : 0.0);

    if (support_sum == 0.0) {
        fprintf(stderr, "No activity in voxels with density >= %g: nothing to do\n", o.support_rho);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_branch, model.br, sizeof(model.br)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_nbranch, &model.n, sizeof(int)));
    if (!o.exact_range) {
        const float scale = kRangeBins / g.cutoff;
        std::vector<float> table(kRangeBins + 1);
        for (int j = 0; j <= kRangeBins; ++j) table[j] = survival(model.br, model.n, j / scale);
        float* d_table = device_alloc<float>(table.size());
        CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_survival, &d_table, sizeof(d_table)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_survival_scale, &scale, sizeof(scale)));
    }

    cudaArray_t rho_array = nullptr;
    const cudaTextureObject_t tex = make_density_texture(rho, g, &rho_array);
    unsigned char* d_support = device_alloc<unsigned char>(n);
    CUDA_CHECK(cudaMemcpy(d_support, support.data(), n, cudaMemcpyHostToDevice));
    float* d_g = device_alloc<float>(n);
    CUDA_CHECK(cudaMemcpy(d_g, img.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    double t = now_s();
    const Projector A(g, tex, active, o.rays, o.cache_near);
    printf("normalisation (%d near-block + %d far-field rays): %.2f s\n", kNearRays, o.rays, now_s() - t);

    float* d_out = device_alloc<float>(n);
    std::vector<float>& out = img;  // input is already on the GPU; reuse its host buffer
    if (o.mode == "forward" || o.mode == "adjoint") {
        t = now_s();
        if (o.mode == "forward") A.forward(d_g, d_out);
        else A.adjoint(d_g, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("%s: %.3f s\n", o.mode.c_str(), now_s() - t);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        write_raw(o.output, out);
        printf("sum in %.6g, sum out %.6g, total %.2f s\n", support_sum, device_sum(d_out, n), now_s() - t0);
        return EXIT_SUCCESS;
    }

    // ---- MAP-EM (Richardson-Lucy + relative-difference prior), initialised with the blurred image and
    // accelerated by Biggs-Andrews extrapolation: each step starts from y = f + lambda (f - prev), where
    // lambda is the correlation of the last two steps h = (new estimate - y), clipped to [0, 1].
    printf("MAP-EM: %d iterations, beta %.4g, gamma %.3g\n", o.iterations, o.beta, o.gamma);
    float* d_f = device_alloc<float>(n);       // estimate
    float* d_prev = d_out;  // previous estimate during prediction, then adjoint scratch
    float* d_y = o.accelerate ? device_alloc<float>(n) : nullptr;  // start, then step h
    float* d_h_prev = o.accelerate ? device_alloc<float>(n) : nullptr;  // previous step
    float* d_est = device_alloc<float>(n);     // A y, then g / A y, then the new estimate
    CUDA_CHECK(cudaMemcpy(d_f, d_g, n * sizeof(float), cudaMemcpyDeviceToDevice));
    const int threads = 256, blocks = (int)((n + threads - 1) / threads);
    const float eps = (float)(1.0e-6 * support_sum / active.size());
    if (o.iterations > 0) {
        initialize_kernel<<<blocks, threads>>>(d_f, d_support, n, 1000.0f * eps);
        finish_kernel<<<blocks, threads>>>(d_f, nullptr, (float)(support_sum / device_sum(d_f, n)), n);
    }
    float lambda = 0.0f;
    t = now_s();
    for (int it = 1; it <= o.iterations; ++it) {
        const float* d_start = d_f;
        if (o.accelerate) {
            extrapolate_kernel<<<blocks, threads>>>(d_f, d_prev, d_y, lambda, n);
            d_start = d_y;
        }
        A.forward(d_start, d_est);
        ratio_kernel<<<blocks, threads>>>(d_g, d_est, d_support, d_est, n, eps);
        // Extrapolation has consumed prev, so its buffer can now hold the adjoint.
        A.adjoint(d_est, d_prev);
        update_kernel<<<blocks, threads>>>(d_start, d_prev, d_support, d_est, g, o.beta, o.gamma, eps);
        CUDA_CHECK(cudaGetLastError());
        float scale = 1.0f;
        if (o.beta > 0.0f) {  // the prior changes the total slightly; RL itself preserves it
            const double s = device_sum(d_est, n);
            if (s > 0) scale = (float)(support_sum / s);
        }
        if (o.beta > 0.0f || o.accelerate)
            finish_kernel<<<blocks, threads>>>(d_est, d_y, scale, n);
        if (o.accelerate) {
            const double hh = device_dot(d_h_prev, d_h_prev, n);
            lambda = hh > 0.0 ? (float)std::min(1.0, std::max(0.0, device_dot(d_y, d_h_prev, n) / hh)) : 0.0f;
            std::swap(d_y, d_h_prev);
            std::swap(d_prev, d_f);  // retain current estimate as prev for the next prediction
        }
        std::swap(d_f, d_est);
        if (it == 1 || it % 10 == 0 || it == o.iterations) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("iteration %d/%d: %.2f s\n", it, o.iterations, now_s() - t);
        }
        if (o.save_every > 0 && it % o.save_every == 0 && it < o.iterations) {
            CUDA_CHECK(cudaMemcpy(out.data(), d_f, n * sizeof(float), cudaMemcpyDeviceToHost));
            char tag[32];
            snprintf(tag, sizeof(tag), "_it%03d", it);
            write_raw(with_suffix(o.output, tag), out);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t_iter = now_s() - t;
    CUDA_CHECK(cudaMemcpy(out.data(), d_f, n * sizeof(float), cudaMemcpyDeviceToHost));
    write_raw(o.output, out);
    if (o.reproject) {
        A.forward(d_f, d_est);
        CUDA_CHECK(cudaMemcpy(out.data(), d_est, n * sizeof(float), cudaMemcpyDeviceToHost));
        write_raw(with_suffix(o.output, "_REPROJ"), out);
    }
    printf("sum blurred (tissue) %.6g, sum corrected %.6g\n", support_sum, device_sum(d_f, n));
    printf("iterations %.2f s (%.2f s each), total %.2f s\n", t_iter, o.iterations ? t_iter / o.iterations : 0.0,
           now_s() - t0);
    return EXIT_SUCCESS;
}
