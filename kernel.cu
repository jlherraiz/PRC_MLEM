#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "cuda_runtime.h"
#include "kernel.h"

#ifndef RAY_NEAR_FIELD_RADIUS
#define RAY_NEAR_FIELD_RADIUS 2
#endif

#ifndef RAY_NEAR_FIELD_SUBDIV
#define RAY_NEAR_FIELD_SUBDIV 2
#endif

namespace {

constexpr int PARAM_SX = 0;
constexpr int PARAM_SY = 1;
constexpr int PARAM_SZ = 2;
constexpr int PARAM_STEP = 3;
constexpr int PARAM_CUTOFF = 4;
constexpr int PARAM_BRANCHES = 5;
constexpr int PARAM_BRANCH_BASE = 6;
constexpr int PARAM_BRANCH_STRIDE = 5;

constexpr int BRANCH_WEIGHT = 0;
constexpr int BRANCH_A = 1;
constexpr int BRANCH_B = 2;
constexpr int BRANCH_C = 3;
constexpr int BRANCH_F = 4;

constexpr int MAX_RAY_TILE = 256;
constexpr float MIN_DENSITY = 1.0e-3f;
constexpr float FOUR_PI = 12.566370614359172f;
constexpr float LARGE_DISTANCE = 1.0e30f;
constexpr int NEAR_FIELD_RADIUS = RAY_NEAR_FIELD_RADIUS;
constexpr int NEAR_FIELD_SIDE = 2 * NEAR_FIELD_RADIUS + 1;
constexpr int NEAR_FIELD_VOXELS = NEAR_FIELD_SIDE * NEAR_FIELD_SIDE * NEAR_FIELD_SIDE;
constexpr int LOW_DENSITY_NEAR_FIELD_RADIUS = 1;
constexpr float LOW_DENSITY_NEAR_FIELD_THRESHOLD = 0.5f;
constexpr int NEAR_FIELD_SUBDIV = RAY_NEAR_FIELD_SUBDIV;
constexpr float NEAR_FIELD_SUBSTEP = 1.0f / static_cast<float>(NEAR_FIELD_SUBDIV);
constexpr float NEAR_FIELD_SUBBASE = -0.5f + 0.5f * NEAR_FIELD_SUBSTEP;

}  // namespace

__device__ float branch_pdf(const float w, const float* par, const int base) {
    const float weight = par[base + BRANCH_WEIGHT];
    const float a = par[base + BRANCH_A];
    const float b = par[base + BRANCH_B];
    const float c = par[base + BRANCH_C];
    const float f = par[base + BRANCH_F];

    const float aw = a * w;
    const float bw = b * w;
    const float cw = c * w;

    float p = powf(aw, 1.0f + f);
    float dp = a * (1.0f + f) * powf(aw, f);

    if (b > 0.0f) {
        p += powf(bw, 2.5f);
        dp += 2.5f * b * powf(bw, 1.5f);
    }
    if (c > 0.0f) {
        p += powf(cw, 3.5f);
        dp += 3.5f * c * powf(cw, 2.5f);
    }

    return weight * dp * expf(-p);
}

__device__ float mixed_pdf(const float w, const float* par, const int branches) {
    float pdf = 0.0f;
    for (int branch = 0; branch < branches; ++branch) {
        const int base = PARAM_BRANCH_BASE + branch * PARAM_BRANCH_STRIDE;
        pdf += branch_pdf(w, par, base);
    }
    return pdf;
}

__device__ float ray_box_exit_mm(const float3 point,
                                 const float sx,
                                 const float sy,
                                 const float sz,
                                 const float half_extent_vox) {
    const float tx = fabsf(point.x) > 1.0e-6f ? half_extent_vox * sx / fabsf(point.x) : LARGE_DISTANCE;
    const float ty = fabsf(point.y) > 1.0e-6f ? half_extent_vox * sy / fabsf(point.y) : LARGE_DISTANCE;
    const float tz = fabsf(point.z) > 1.0e-6f ? half_extent_vox * sz / fabsf(point.z) : LARGE_DISTANCE;
    return fminf(tx, fminf(ty, tz));
}

__device__ float ray_source_exit_mm(const float3 point, const float sx, const float sy, const float sz) {
    return ray_box_exit_mm(point, sx, sy, sz, 0.5f);
}

__device__ int ray_near_field_radius_for_density(const float rho_source) {
    if (NEAR_FIELD_RADIUS <= LOW_DENSITY_NEAR_FIELD_RADIUS) {
        return NEAR_FIELD_RADIUS;
    }
    return rho_source < LOW_DENSITY_NEAR_FIELD_THRESHOLD ? LOW_DENSITY_NEAR_FIELD_RADIUS : NEAR_FIELD_RADIUS;
}

__device__ bool ray_inside_near_field(const int idx, const int idy, const int idz, const int active_radius) {
    return max(abs(idx), max(abs(idy), abs(idz))) <= active_radius;
}

__device__ float ray_near_field_exit_mm(const float3 point,
                                        const float sx,
                                        const float sy,
                                        const float sz,
                                        const int active_radius) {
    return ray_box_exit_mm(point, sx, sy, sz, static_cast<float>(active_radius) + 0.5f);
}

__device__ float ray_integrate_weq_to_distance(cudaTextureObject_t DENS_tex,
                                               const float3 point,
                                               const int ix,
                                               const int iy,
                                               const int iz,
                                               const float s_end_mm,
                                               const float sx,
                                               const float sy,
                                               const float sz,
                                               const float step_mm,
                                               const int RESX,
                                               const int RESY,
                                               const int NZS) {
    if (!(s_end_mm > 0.0f)) {
        return 0.0f;
    }

    const int nsamp = max(1, static_cast<int>(ceilf(s_end_mm / step_mm)));
    const float ds_mm = s_end_mm / static_cast<float>(nsamp);
    float weq = 0.0f;

    for (int isamp = 0; isamp < nsamp; ++isamp) {
        const float s_mm = (static_cast<float>(isamp) + 0.5f) * ds_mm;
        const float x = ix + (s_mm * point.x) / sx;
        const float y = iy + (s_mm * point.y) / sy;
        const float z = iz + (s_mm * point.z) / sz;

        if (x < 0.0f || x >= RESX || y < 0.0f || y >= RESY || z < 0.0f || z >= NZS) {
            break;
        }

        const float rho = fmaxf(tex3D<float>(DENS_tex, x + 0.5f, y + 0.5f, z + 0.5f), MIN_DENSITY);
        weq += rho * ds_mm;
    }

    return weq;
}

__device__ float ray_integrate_weq_to_offset(cudaTextureObject_t DENS_tex,
                                             const int ix,
                                             const int iy,
                                             const int iz,
                                             const float offx,
                                             const float offy,
                                             const float offz,
                                             const float sx,
                                             const float sy,
                                             const float sz,
                                             const float step_mm) {
    const float dx_mm = offx * sx;
    const float dy_mm = offy * sy;
    const float dz_mm = offz * sz;
    const float dist_mm = sqrtf(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm);

    if (!(dist_mm > 0.0f)) {
        return 0.0f;
    }

    const int nsamp = max(1, static_cast<int>(ceilf(dist_mm / step_mm)));
    const float ds_mm = dist_mm / static_cast<float>(nsamp);
    float weq = 0.0f;

    for (int isamp = 0; isamp < nsamp; ++isamp) {
        const float lambda = (static_cast<float>(isamp) + 0.5f) / static_cast<float>(nsamp);
        const float x = ix + offx * lambda;
        const float y = iy + offy * lambda;
        const float z = iz + offz * lambda;
        const float rho = fmaxf(tex3D<float>(DENS_tex, x + 0.5f, y + 0.5f, z + 0.5f), MIN_DENSITY);
        weq += rho * ds_mm;
    }

    return weq;
}

__device__ void near_field_offset_from_linear(const int linear_idx,
                                              int* idx,
                                              int* idy,
                                              int* idz) {
    *idx = linear_idx % NEAR_FIELD_SIDE - NEAR_FIELD_RADIUS;
    *idy = (linear_idx / NEAR_FIELD_SIDE) % NEAR_FIELD_SIDE - NEAR_FIELD_RADIUS;
    *idz = linear_idx / (NEAR_FIELD_SIDE * NEAR_FIELD_SIDE) - NEAR_FIELD_RADIUS;
}

__device__ float ray_local_volume_density(cudaTextureObject_t DENS_tex,
                                          const float* Par,
                                          const int branches,
                                          const int ix,
                                          const int iy,
                                          const int iz,
                                          const float offx,
                                          const float offy,
                                          const float offz) {
    const float SX = Par[PARAM_SX];
    const float SY = Par[PARAM_SY];
    const float SZ = Par[PARAM_SZ];
    const float step_mm = Par[PARAM_STEP];
    const float cutoff_mm = Par[PARAM_CUTOFF];
    const float dx_mm = offx * SX;
    const float dy_mm = offy * SY;
    const float dz_mm = offz * SZ;
    const float s_mm = sqrtf(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm);

    if (!(s_mm > 1.0e-6f)) {
        return 0.0f;
    }

    const float w_center = ray_integrate_weq_to_offset(DENS_tex, ix, iy, iz, offx, offy, offz, SX, SY, SZ, step_mm);
    if (w_center > cutoff_mm) {
        return 0.0f;
    }

    const float rho_local =
        fmaxf(tex3D<float>(DENS_tex, ix + offx + 0.5f, iy + offy + 0.5f, iz + offz + 0.5f), MIN_DENSITY);
    const float density = mixed_pdf(w_center, Par, branches) * rho_local / (FOUR_PI * s_mm * s_mm);
    return (density > 0.0f && isfinite(density)) ? density : 0.0f;
}

__device__ float ray_local_voxel_probability(cudaTextureObject_t DENS_tex,
                                             const float* Par,
                                             const int branches,
                                             const int ix,
                                             const int iy,
                                             const int iz,
                                             const int idx,
                                             const int idy,
                                             const int idz) {
    const float SX = Par[PARAM_SX];
    const float SY = Par[PARAM_SY];
    const float SZ = Par[PARAM_SZ];
    const float sub_volume_mm3 = (SX * SY * SZ) / static_cast<float>(NEAR_FIELD_SUBDIV * NEAR_FIELD_SUBDIV *
                                                                      NEAR_FIELD_SUBDIV);

    float probability = 0.0f;

    for (int qz = 0; qz < NEAR_FIELD_SUBDIV; ++qz) {
        const float offz = static_cast<float>(idz) + NEAR_FIELD_SUBBASE + static_cast<float>(qz) * NEAR_FIELD_SUBSTEP;
        for (int qy = 0; qy < NEAR_FIELD_SUBDIV; ++qy) {
            const float offy =
                static_cast<float>(idy) + NEAR_FIELD_SUBBASE + static_cast<float>(qy) * NEAR_FIELD_SUBSTEP;
            for (int qx = 0; qx < NEAR_FIELD_SUBDIV; ++qx) {
                const float offx =
                    static_cast<float>(idx) + NEAR_FIELD_SUBBASE + static_cast<float>(qx) * NEAR_FIELD_SUBSTEP;
                probability += ray_local_volume_density(DENS_tex, Par, branches, ix, iy, iz, offx, offy, offz) *
                               sub_volume_mm3;
            }
        }
    }

    return (probability > 0.0f && isfinite(probability)) ? probability : 0.0f;
}

__device__ float ray_source_voxel_probability(cudaTextureObject_t DENS_tex,
                                              const float* Par,
                                              const int branches,
                                              const int ix,
                                              const int iy,
                                              const int iz) {
    return ray_local_voxel_probability(DENS_tex, Par, branches, ix, iy, iz, 0, 0, 0);
}

__device__ float ray_near_field_target_probability(cudaTextureObject_t DENS_tex,
                                                   const float* Par,
                                                   const int branches,
                                                   const int ix,
                                                   const int iy,
                                                   const int iz,
                                                   const int idx,
                                                   const int idy,
                                                   const int idz) {
    if (idx == 0 && idy == 0 && idz == 0) {
        return 0.0f;
    }
    return ray_local_voxel_probability(DENS_tex, Par, branches, ix, iy, iz, idx, idy, idz);
}

__device__ void ray_near_field_prefix(cudaTextureObject_t DENS_tex,
                                      const float3 point,
                                      const float* Par,
                                      const int active_radius,
                                      const int ix,
                                      const int iy,
                                      const int iz,
                                      const int RESX,
                                      const int RESY,
                                      const int NZS,
                                      float* s_exit_mm,
                                      float* weq_exit) {
    const float SX = Par[PARAM_SX];
    const float SY = Par[PARAM_SY];
    const float SZ = Par[PARAM_SZ];
    const float step_mm = Par[PARAM_STEP];

    *s_exit_mm = ray_near_field_exit_mm(point, SX, SY, SZ, active_radius);
    *weq_exit = ray_integrate_weq_to_distance(DENS_tex, point, ix, iy, iz, *s_exit_mm, SX, SY, SZ, step_mm, RESX,
                                              RESY, NZS);
}

__device__ float ray_total_weight_from(cudaTextureObject_t DENS_tex,
                                       const float3 point,
                                       const float* Par,
                                       const int ix,
                                       const int iy,
                                       const int iz,
                                       const int RESX,
                                       const int RESY,
                                       const int NZS,
                                       const float s_start_mm,
                                       float weq_acc) {
    const float SX = Par[PARAM_SX];
    const float SY = Par[PARAM_SY];
    const float SZ = Par[PARAM_SZ];
    const float cutoff_mm = Par[PARAM_CUTOFF];
    const int branches = static_cast<int>(Par[PARAM_BRANCHES]);

    if (weq_acc >= cutoff_mm) {
        return 0.0f;
    }

    const float eps_mm = 1.0e-4f * fminf(SX, fminf(SY, SZ));
    float s_mm = s_start_mm + eps_mm;
    int i = static_cast<int>(floorf(ix + (s_mm * point.x) / SX + 0.5f));
    int j = static_cast<int>(floorf(iy + (s_mm * point.y) / SY + 0.5f));
    int k = static_cast<int>(floorf(iz + (s_mm * point.z) / SZ + 0.5f));

    if (i < 0 || i >= RESX || j < 0 || j >= RESY || k < 0 || k >= NZS) {
        return 0.0f;
    }

    const int stepx = point.x > 1.0e-6f ? 1 : (point.x < -1.0e-6f ? -1 : 0);
    const int stepy = point.y > 1.0e-6f ? 1 : (point.y < -1.0e-6f ? -1 : 0);
    const int stepz = point.z > 1.0e-6f ? 1 : (point.z < -1.0e-6f ? -1 : 0);

    const float x0 = ix + (s_mm * point.x) / SX;
    const float y0 = iy + (s_mm * point.y) / SY;
    const float z0 = iz + (s_mm * point.z) / SZ;

    float tmax_x = LARGE_DISTANCE;
    float tmax_y = LARGE_DISTANCE;
    float tmax_z = LARGE_DISTANCE;
    float tdelta_x = LARGE_DISTANCE;
    float tdelta_y = LARGE_DISTANCE;
    float tdelta_z = LARGE_DISTANCE;

    if (stepx != 0) {
        const float boundary_x = static_cast<float>(i) + 0.5f * static_cast<float>(stepx);
        tmax_x = s_mm + (boundary_x - x0) * SX / point.x;
        tdelta_x = SX / fabsf(point.x);
    }
    if (stepy != 0) {
        const float boundary_y = static_cast<float>(j) + 0.5f * static_cast<float>(stepy);
        tmax_y = s_mm + (boundary_y - y0) * SY / point.y;
        tdelta_y = SY / fabsf(point.y);
    }
    if (stepz != 0) {
        const float boundary_z = static_cast<float>(k) + 0.5f * static_cast<float>(stepz);
        tmax_z = s_mm + (boundary_z - z0) * SZ / point.z;
        tdelta_z = SZ / fabsf(point.z);
    }

    float ray_sum = 0.0f;

    while (i >= 0 && i < RESX && j >= 0 && j < RESY && k >= 0 && k < NZS) {
        float s_next = fminf(tmax_x, fminf(tmax_y, tmax_z));
        float ds_mm = s_next - s_mm;

        if (!(ds_mm > 1.0e-6f)) {
            if (fabsf(tmax_x - s_next) <= 1.0e-6f) {
                i += stepx;
                tmax_x += tdelta_x;
            }
            if (fabsf(tmax_y - s_next) <= 1.0e-6f) {
                j += stepy;
                tmax_y += tdelta_y;
            }
            if (fabsf(tmax_z - s_next) <= 1.0e-6f) {
                k += stepz;
                tmax_z += tdelta_z;
            }
            s_mm = s_next;
            continue;
        }

        const float rho = fmaxf(tex3D<float>(DENS_tex, i + 0.5f, j + 0.5f, k + 0.5f), MIN_DENSITY);
        const float remaining_weq = cutoff_mm - weq_acc;
        if (!(remaining_weq > 0.0f)) {
            break;
        }

        float used_ds_mm = ds_mm;
        float used_dw = rho * used_ds_mm;
        if (used_dw > remaining_weq) {
            used_dw = remaining_weq;
            used_ds_mm = used_dw / rho;
        }

        if (!(used_dw > 0.0f)) {
            break;
        }

        const float weq_mid = weq_acc + 0.5f * used_dw;
        ray_sum += mixed_pdf(weq_mid, Par, branches) * used_dw;
        weq_acc += used_dw;

        if (used_ds_mm < ds_mm - 1.0e-6f) {
            break;
        }

        s_mm = s_next;
        if (fabsf(tmax_x - s_next) <= 1.0e-6f) {
            i += stepx;
            tmax_x += tdelta_x;
        }
        if (fabsf(tmax_y - s_next) <= 1.0e-6f) {
            j += stepy;
            tmax_y += tdelta_y;
        }
        if (fabsf(tmax_z - s_next) <= 1.0e-6f) {
            k += stepz;
            tmax_z += tdelta_z;
        }
    }

    return ray_sum;
}

__device__ void deposit_ray_deterministic_from(cudaTextureObject_t DENS_tex,
                                               float* PRC,
                                               const float PETV,
                                               const float total_weight,
                                               const float3 point,
                                               const float* Par,
                                               const int ix,
                                               const int iy,
                                               const int iz,
                                               const int RESX,
                                               const int RESY,
                                               const int NZS,
                                               const float s_start_mm,
                                               float weq_acc) {
    if (!(total_weight > 0.0f) || !(PETV > 0.0f)) {
        return;
    }

    const float SX = Par[PARAM_SX];
    const float SY = Par[PARAM_SY];
    const float SZ = Par[PARAM_SZ];
    const float cutoff_mm = Par[PARAM_CUTOFF];
    const int branches = static_cast<int>(Par[PARAM_BRANCHES]);

    if (weq_acc >= cutoff_mm) {
        return;
    }

    const float eps_mm = 1.0e-4f * fminf(SX, fminf(SY, SZ));
    float s_mm = s_start_mm + eps_mm;
    int i = static_cast<int>(floorf(ix + (s_mm * point.x) / SX + 0.5f));
    int j = static_cast<int>(floorf(iy + (s_mm * point.y) / SY + 0.5f));
    int k = static_cast<int>(floorf(iz + (s_mm * point.z) / SZ + 0.5f));

    if (i < 0 || i >= RESX || j < 0 || j >= RESY || k < 0 || k >= NZS) {
        return;
    }

    const int stepx = point.x > 1.0e-6f ? 1 : (point.x < -1.0e-6f ? -1 : 0);
    const int stepy = point.y > 1.0e-6f ? 1 : (point.y < -1.0e-6f ? -1 : 0);
    const int stepz = point.z > 1.0e-6f ? 1 : (point.z < -1.0e-6f ? -1 : 0);

    const float x0 = ix + (s_mm * point.x) / SX;
    const float y0 = iy + (s_mm * point.y) / SY;
    const float z0 = iz + (s_mm * point.z) / SZ;

    float tmax_x = LARGE_DISTANCE;
    float tmax_y = LARGE_DISTANCE;
    float tmax_z = LARGE_DISTANCE;
    float tdelta_x = LARGE_DISTANCE;
    float tdelta_y = LARGE_DISTANCE;
    float tdelta_z = LARGE_DISTANCE;

    if (stepx != 0) {
        const float boundary_x = static_cast<float>(i) + 0.5f * static_cast<float>(stepx);
        tmax_x = s_mm + (boundary_x - x0) * SX / point.x;
        tdelta_x = SX / fabsf(point.x);
    }
    if (stepy != 0) {
        const float boundary_y = static_cast<float>(j) + 0.5f * static_cast<float>(stepy);
        tmax_y = s_mm + (boundary_y - y0) * SY / point.y;
        tdelta_y = SY / fabsf(point.y);
    }
    if (stepz != 0) {
        const float boundary_z = static_cast<float>(k) + 0.5f * static_cast<float>(stepz);
        tmax_z = s_mm + (boundary_z - z0) * SZ / point.z;
        tdelta_z = SZ / fabsf(point.z);
    }

    while (i >= 0 && i < RESX && j >= 0 && j < RESY && k >= 0 && k < NZS) {
        float s_next = fminf(tmax_x, fminf(tmax_y, tmax_z));
        float ds_mm = s_next - s_mm;

        if (!(ds_mm > 1.0e-6f)) {
            if (fabsf(tmax_x - s_next) <= 1.0e-6f) {
                i += stepx;
                tmax_x += tdelta_x;
            }
            if (fabsf(tmax_y - s_next) <= 1.0e-6f) {
                j += stepy;
                tmax_y += tdelta_y;
            }
            if (fabsf(tmax_z - s_next) <= 1.0e-6f) {
                k += stepz;
                tmax_z += tdelta_z;
            }
            s_mm = s_next;
            continue;
        }

        const float rho = fmaxf(tex3D<float>(DENS_tex, i + 0.5f, j + 0.5f, k + 0.5f), MIN_DENSITY);
        const float remaining_weq = cutoff_mm - weq_acc;
        if (!(remaining_weq > 0.0f)) {
            break;
        }

        float used_ds_mm = ds_mm;
        float used_dw = rho * used_ds_mm;
        if (used_dw > remaining_weq) {
            used_dw = remaining_weq;
            used_ds_mm = used_dw / rho;
        }

        if (!(used_dw > 0.0f)) {
            break;
        }

        const float weq_mid = weq_acc + 0.5f * used_dw;
        const float weight = mixed_pdf(weq_mid, Par, branches) * used_dw;
        if (weight > 0.0f) {
            const int iVV = k * RESX * RESY + j * RESX + i;
            atomicAdd(&PRC[iVV], PETV * weight / total_weight);
        }

        weq_acc += used_dw;
        if (used_ds_mm < ds_mm - 1.0e-6f) {
            break;
        }

        s_mm = s_next;
        if (fabsf(tmax_x - s_next) <= 1.0e-6f) {
            i += stepx;
            tmax_x += tdelta_x;
        }
        if (fabsf(tmax_y - s_next) <= 1.0e-6f) {
            j += stepy;
            tmax_y += tdelta_y;
        }
        if (fabsf(tmax_z - s_next) <= 1.0e-6f) {
            k += stepz;
            tmax_z += tdelta_z;
        }
    }
}

__global__ void POSITRONRANGEKERNEL(cudaTextureObject_t DENS_tex,
                                    float* PRC,
                                    const float* PET,
                                    const float3* POINTS,
                                    const float* Par,
                                    const int N_RAYS) {
    __shared__ float tile_reduce[MAX_RAY_TILE];
    __shared__ float near_weights[NEAR_FIELD_VOXELS];
    __shared__ int near_targets[NEAR_FIELD_VOXELS];
    __shared__ float total_weight_shared;
    __shared__ float source_weight_shared;
    __shared__ float near_weight_shared;
    __shared__ float source_rho_shared;
    __shared__ int active_near_field_radius_shared;

    const int ix = blockIdx.x;
    const int iy = blockIdx.y;
    const int iz = blockIdx.z;
    const int tid = threadIdx.x;

    if (blockDim.x > MAX_RAY_TILE) {
        return;
    }

    const int RESX = gridDim.x;
    const int RESY = gridDim.y;
    const int NZS = gridDim.z;
    const int iV = iz * RESX * RESY + iy * RESX + ix;
    const float PETV = PET[iV];
    const int branches = static_cast<int>(Par[PARAM_BRANCHES]);

    if (PETV <= 0.0f) {
        return;
    }

    if (tid == 0) {
        total_weight_shared = 0.0f;
        near_weight_shared = 0.0f;
        source_rho_shared = fmaxf(tex3D<float>(DENS_tex, ix + 0.5f, iy + 0.5f, iz + 0.5f), MIN_DENSITY);
        active_near_field_radius_shared = ray_near_field_radius_for_density(source_rho_shared);
        source_weight_shared =
            static_cast<float>(N_RAYS) * ray_source_voxel_probability(DENS_tex, Par, branches, ix, iy, iz);
    }
    __syncthreads();

    float near_weight_sum = 0.0f;
    for (int near_idx = tid; near_idx < NEAR_FIELD_VOXELS; near_idx += blockDim.x) {
        int idx = 0;
        int idy = 0;
        int idz = 0;
        near_field_offset_from_linear(near_idx, &idx, &idy, &idz);
        const int i = ix + idx;
        const int j = iy + idy;
        const int k = iz + idz;
        float near_weight = 0.0f;
        int near_target = -1;
        if (ray_inside_near_field(idx, idy, idz, active_near_field_radius_shared) && !(idx == 0 && idy == 0 && idz == 0) &&
            i >= 0 && i < RESX && j >= 0 && j < RESY && k >= 0 &&
            k < NZS) {
            near_target = k * RESX * RESY + j * RESX + i;
            near_weight =
                static_cast<float>(N_RAYS) * ray_near_field_target_probability(DENS_tex, Par, branches, ix, iy, iz,
                                                                                idx, idy, idz);
        }
        near_weights[near_idx] = near_weight;
        near_targets[near_idx] = near_target;
        near_weight_sum += near_weight;
    }
    tile_reduce[tid] = near_weight_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tile_reduce[tid] += tile_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        near_weight_shared = tile_reduce[0];
        total_weight_shared = near_weight_shared + source_weight_shared;
    }
    __syncthreads();

    for (int ray_offset = 0; ray_offset < N_RAYS; ray_offset += blockDim.x) {
        const int ray_idx = ray_offset + tid;
        float ray_sum = 0.0f;
        if (ray_idx < N_RAYS) {
            float near_exit_mm = 0.0f;
            float near_weq_exit = 0.0f;
            ray_near_field_prefix(DENS_tex, POINTS[ray_idx], Par, active_near_field_radius_shared, ix, iy, iz, RESX,
                                  RESY, NZS, &near_exit_mm, &near_weq_exit);
            ray_sum = ray_total_weight_from(DENS_tex, POINTS[ray_idx], Par, ix, iy, iz, RESX, RESY, NZS,
                                            near_exit_mm, near_weq_exit);
        }

        tile_reduce[tid] = ray_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                tile_reduce[tid] += tile_reduce[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            total_weight_shared += tile_reduce[0];
        }
        __syncthreads();
    }

    if (total_weight_shared <= 0.0f) {
        return;
    }

    if (tid == 0 && source_weight_shared > 0.0f) {
        atomicAdd(&PRC[iV], PETV * source_weight_shared / total_weight_shared);
    }
    __syncthreads();

    for (int near_idx = tid; near_idx < NEAR_FIELD_VOXELS; near_idx += blockDim.x) {
        if (near_targets[near_idx] >= 0 && near_weights[near_idx] > 0.0f) {
            atomicAdd(&PRC[near_targets[near_idx]], PETV * near_weights[near_idx] / total_weight_shared);
        }
    }
    __syncthreads();

    for (int ray_offset = 0; ray_offset < N_RAYS; ray_offset += blockDim.x) {
        const int ray_idx = ray_offset + tid;
        if (ray_idx < N_RAYS) {
            float near_exit_mm = 0.0f;
            float near_weq_exit = 0.0f;
            ray_near_field_prefix(DENS_tex, POINTS[ray_idx], Par, active_near_field_radius_shared, ix, iy, iz, RESX,
                                  RESY, NZS, &near_exit_mm, &near_weq_exit);
            deposit_ray_deterministic_from(DENS_tex, PRC, PETV, total_weight_shared, POINTS[ray_idx], Par, ix, iy,
                                           iz, RESX, RESY, NZS, near_exit_mm, near_weq_exit);
        }
        __syncthreads();
    }
}

int choose_ray_tile_size(const int nrays) {
    int threads = 32;
    while (threads < nrays && threads < MAX_RAY_TILE) {
        threads <<= 1;
    }
    if (threads > MAX_RAY_TILE) {
        threads = MAX_RAY_TILE;
    }
    return threads;
}

void POSITRONRANGE_EX(cudaTextureObject_t DENS_tex,
                      float* d_POSITRON,
                      const float* d_PET,
                      const float3* d_POINTS,
                      const float* d_PARAMETERS,
                      int RESX,
                      int RESY,
                      int NZS,
                      int N,
                      bool verbose) {
    const int threads = choose_ray_tile_size(N);
    if (verbose) {
        printf("RX = %i, RY = %i, RZ = %i, RAYS = %i, RAY_TILE = %i\n", RESX, RESY, NZS, N, threads);
    }
    dim3 threadsPerBlock(threads);
    dim3 blocksPerGrid(RESX, RESY, NZS);
    POSITRONRANGEKERNEL<<<blocksPerGrid, threadsPerBlock>>>(DENS_tex, d_POSITRON, d_PET, d_POINTS,
                                                            d_PARAMETERS, N);
}

void POSITRONRANGE(cudaTextureObject_t DENS_tex,
                   float* d_POSITRON,
                   const float* d_PET,
                   const float3* d_POINTS,
                   const float* d_PARAMETERS,
                   int RESX,
                   int RESY,
                   int NZS,
                   int N) {
    POSITRONRANGE_EX(DENS_tex, d_POSITRON, d_PET, d_POINTS, d_PARAMETERS, RESX, RESY, NZS, N, true);
}

cudaTextureObject_t create_texture_density(const float* h_imagen, int RESX, int RESY, int NZS) {
    cudaArray* array3d_DENS = nullptr;
    const cudaExtent dims = make_cudaExtent(RESX, RESY, NZS);
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaError_t err = cudaMalloc3DArray(&array3d_DENS, &desc, dims);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc3DArray failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy3DParms params = {};
    params.srcPtr = make_cudaPitchedPtr((void*)h_imagen, RESX * sizeof(float), RESX, RESY);
    params.dstArray = array3d_DENS;
    params.extent = dims;
    params.kind = cudaMemcpyHostToDevice;

    err = cudaMemcpy3D(&params);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3D failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array3d_DENS;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t DENS_tex = 0;
    err = cudaCreateTextureObject(&DENS_tex, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaCreateTextureObject failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return DENS_tex;
}

void chequearError(const char* errorStr) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%s: %s.\n", errorStr, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("%s: %s.\n", errorStr, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
