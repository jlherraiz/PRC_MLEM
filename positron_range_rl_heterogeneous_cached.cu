#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"
#include "positron_range_host_common.hpp"

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

constexpr float MIN_DENSITY = 1.0e-3f;
constexpr float LARGE_DISTANCE = 1.0e30f;
constexpr float FOUR_PI = 12.566370614359172f;
constexpr int NEAR_FIELD_RADIUS = RAY_NEAR_FIELD_RADIUS;
constexpr int LOW_DENSITY_NEAR_FIELD_RADIUS = 1;
constexpr float LOW_DENSITY_NEAR_FIELD_THRESHOLD = 0.5f;
constexpr int NEAR_FIELD_SUBDIV = RAY_NEAR_FIELD_SUBDIV;
constexpr float NEAR_FIELD_SUBSTEP = 1.0f / static_cast<float>(NEAR_FIELD_SUBDIV);
constexpr float NEAR_FIELD_SUBBASE = -0.5f + 0.5f * NEAR_FIELD_SUBSTEP;

struct TileCache {
    std::vector<int> source_ids;
    std::vector<int> row_offsets;
    std::vector<int> target_indices;
    std::vector<float> weights;
};

inline int voxel_index(const int x, const int y, const int z, const int resx, const int resy) {
    return z * resx * resy + y * resx + x;
}

inline float clampf_host(const float value, const float lo, const float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

inline float sample_density_linear(const std::vector<float>& density,
                                   const int resx,
                                   const int resy,
                                   const int nzs,
                                   const float x,
                                   const float y,
                                   const float z) {
    const float xf = clampf_host(x - 0.5f, 0.0f, static_cast<float>(resx - 1));
    const float yf = clampf_host(y - 0.5f, 0.0f, static_cast<float>(resy - 1));
    const float zf = clampf_host(z - 0.5f, 0.0f, static_cast<float>(nzs - 1));

    const int x0 = static_cast<int>(floorf(xf));
    const int y0 = static_cast<int>(floorf(yf));
    const int z0 = static_cast<int>(floorf(zf));
    const int x1 = std::min(x0 + 1, resx - 1);
    const int y1 = std::min(y0 + 1, resy - 1);
    const int z1 = std::min(z0 + 1, nzs - 1);

    const float tx = xf - static_cast<float>(x0);
    const float ty = yf - static_cast<float>(y0);
    const float tz = zf - static_cast<float>(z0);

    const auto at = [&](const int xi, const int yi, const int zi) {
        return density[voxel_index(xi, yi, zi, resx, resy)];
    };

    const float c000 = at(x0, y0, z0);
    const float c100 = at(x1, y0, z0);
    const float c010 = at(x0, y1, z0);
    const float c110 = at(x1, y1, z0);
    const float c001 = at(x0, y0, z1);
    const float c101 = at(x1, y0, z1);
    const float c011 = at(x0, y1, z1);
    const float c111 = at(x1, y1, z1);

    const float c00 = c000 + tx * (c100 - c000);
    const float c10 = c010 + tx * (c110 - c010);
    const float c01 = c001 + tx * (c101 - c001);
    const float c11 = c011 + tx * (c111 - c011);

    const float c0 = c00 + ty * (c10 - c00);
    const float c1 = c01 + ty * (c11 - c01);
    return c0 + tz * (c1 - c0);
}

inline float branch_pdf_host(const float w, const std::vector<float>& par, const int base) {
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

inline float mixed_pdf_host(const float w, const std::vector<float>& par) {
    const int branches = static_cast<int>(par[PARAM_BRANCHES]);
    float pdf = 0.0f;
    for (int branch = 0; branch < branches; ++branch) {
        const int base = PARAM_BRANCH_BASE + branch * PARAM_BRANCH_STRIDE;
        pdf += branch_pdf_host(w, par, base);
    }
    return pdf;
}

inline float ray_box_exit_mm_host(const float3 point,
                                  const float sx,
                                  const float sy,
                                  const float sz,
                                  const float half_extent_vox) {
    const float tx = fabsf(point.x) > 1.0e-6f ? half_extent_vox * sx / fabsf(point.x) : LARGE_DISTANCE;
    const float ty = fabsf(point.y) > 1.0e-6f ? half_extent_vox * sy / fabsf(point.y) : LARGE_DISTANCE;
    const float tz = fabsf(point.z) > 1.0e-6f ? half_extent_vox * sz / fabsf(point.z) : LARGE_DISTANCE;
    return std::min(tx, std::min(ty, tz));
}

inline int near_field_radius_for_density_host(const float rho_source) {
    if (NEAR_FIELD_RADIUS <= LOW_DENSITY_NEAR_FIELD_RADIUS) {
        return NEAR_FIELD_RADIUS;
    }
    return rho_source < LOW_DENSITY_NEAR_FIELD_THRESHOLD ? LOW_DENSITY_NEAR_FIELD_RADIUS : NEAR_FIELD_RADIUS;
}

inline bool inside_near_field_host(const int idx, const int idy, const int idz, const int active_radius) {
    return std::max(abs(idx), std::max(abs(idy), abs(idz))) <= active_radius;
}

inline float integrate_weq_to_distance_host(const std::vector<float>& density,
                                            const float3 point,
                                            const int ix,
                                            const int iy,
                                            const int iz,
                                            const float s_end_mm,
                                            const float sx,
                                            const float sy,
                                            const float sz,
                                            const float step_mm,
                                            const int resx,
                                            const int resy,
                                            const int nzs) {
    if (!(s_end_mm > 0.0f)) {
        return 0.0f;
    }

    const int nsamp = std::max(1, static_cast<int>(ceilf(s_end_mm / step_mm)));
    const float ds_mm = s_end_mm / static_cast<float>(nsamp);
    float weq = 0.0f;

    for (int isamp = 0; isamp < nsamp; ++isamp) {
        const float s_mm = (static_cast<float>(isamp) + 0.5f) * ds_mm;
        const float x = ix + (s_mm * point.x) / sx;
        const float y = iy + (s_mm * point.y) / sy;
        const float z = iz + (s_mm * point.z) / sz;

        if (x < 0.0f || x >= resx || y < 0.0f || y >= resy || z < 0.0f || z >= nzs) {
            break;
        }

        const float rho = std::max(sample_density_linear(density, resx, resy, nzs, x + 0.5f, y + 0.5f, z + 0.5f),
                                   MIN_DENSITY);
        weq += rho * ds_mm;
    }

    return weq;
}

inline float integrate_weq_to_offset_host(const std::vector<float>& density,
                                          const int ix,
                                          const int iy,
                                          const int iz,
                                          const float offx,
                                          const float offy,
                                          const float offz,
                                          const float sx,
                                          const float sy,
                                          const float sz,
                                          const float step_mm,
                                          const int resx,
                                          const int resy,
                                          const int nzs) {
    const float dx_mm = offx * sx;
    const float dy_mm = offy * sy;
    const float dz_mm = offz * sz;
    const float dist_mm = sqrtf(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm);

    if (!(dist_mm > 0.0f)) {
        return 0.0f;
    }

    const int nsamp = std::max(1, static_cast<int>(ceilf(dist_mm / step_mm)));
    const float ds_mm = dist_mm / static_cast<float>(nsamp);
    float weq = 0.0f;

    for (int isamp = 0; isamp < nsamp; ++isamp) {
        const float lambda = (static_cast<float>(isamp) + 0.5f) / static_cast<float>(nsamp);
        const float x = ix + offx * lambda;
        const float y = iy + offy * lambda;
        const float z = iz + offz * lambda;
        const float rho = std::max(sample_density_linear(density, resx, resy, nzs, x + 0.5f, y + 0.5f, z + 0.5f),
                                   MIN_DENSITY);
        weq += rho * ds_mm;
    }

    return weq;
}

inline float local_volume_density_host(const std::vector<float>& density,
                                       const std::vector<float>& par,
                                       const int ix,
                                       const int iy,
                                       const int iz,
                                       const float offx,
                                       const float offy,
                                       const float offz,
                                       const int resx,
                                       const int resy,
                                       const int nzs) {
    const float sx = par[PARAM_SX];
    const float sy = par[PARAM_SY];
    const float sz = par[PARAM_SZ];
    const float step_mm = par[PARAM_STEP];
    const float cutoff_mm = par[PARAM_CUTOFF];
    const float dx_mm = offx * sx;
    const float dy_mm = offy * sy;
    const float dz_mm = offz * sz;
    const float s_mm = sqrtf(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm);

    if (!(s_mm > 1.0e-6f)) {
        return 0.0f;
    }

    const float w_center =
        integrate_weq_to_offset_host(density, ix, iy, iz, offx, offy, offz, sx, sy, sz, step_mm, resx, resy, nzs);
    if (w_center > cutoff_mm) {
        return 0.0f;
    }

    const float rho_local = std::max(
        sample_density_linear(density, resx, resy, nzs, ix + offx + 0.5f, iy + offy + 0.5f, iz + offz + 0.5f),
        MIN_DENSITY);
    const float dens = mixed_pdf_host(w_center, par) * rho_local / (FOUR_PI * s_mm * s_mm);
    return (dens > 0.0f && std::isfinite(dens)) ? dens : 0.0f;
}

inline float local_voxel_probability_host(const std::vector<float>& density,
                                          const std::vector<float>& par,
                                          const int ix,
                                          const int iy,
                                          const int iz,
                                          const int idx,
                                          const int idy,
                                          const int idz,
                                          const int resx,
                                          const int resy,
                                          const int nzs) {
    const float sx = par[PARAM_SX];
    const float sy = par[PARAM_SY];
    const float sz = par[PARAM_SZ];
    const float sub_volume_mm3 = (sx * sy * sz) /
                                 static_cast<float>(NEAR_FIELD_SUBDIV * NEAR_FIELD_SUBDIV * NEAR_FIELD_SUBDIV);
    float probability = 0.0f;

    for (int qz = 0; qz < NEAR_FIELD_SUBDIV; ++qz) {
        const float offz = static_cast<float>(idz) + NEAR_FIELD_SUBBASE + static_cast<float>(qz) * NEAR_FIELD_SUBSTEP;
        for (int qy = 0; qy < NEAR_FIELD_SUBDIV; ++qy) {
            const float offy =
                static_cast<float>(idy) + NEAR_FIELD_SUBBASE + static_cast<float>(qy) * NEAR_FIELD_SUBSTEP;
            for (int qx = 0; qx < NEAR_FIELD_SUBDIV; ++qx) {
                const float offx =
                    static_cast<float>(idx) + NEAR_FIELD_SUBBASE + static_cast<float>(qx) * NEAR_FIELD_SUBSTEP;
                probability += local_volume_density_host(density, par, ix, iy, iz, offx, offy, offz, resx, resy, nzs) *
                               sub_volume_mm3;
            }
        }
    }

    return (probability > 0.0f && std::isfinite(probability)) ? probability : 0.0f;
}

inline void add_ray_weights_host(const std::vector<float>& density,
                                 const std::vector<float>& par,
                                 const float3 point,
                                 const int ix,
                                 const int iy,
                                 const int iz,
                                 const int resx,
                                 const int resy,
                                 const int nzs,
                                 const float s_start_mm,
                                 float weq_acc,
                                 std::unordered_map<int, float>* weights,
                                 float* total_weight) {
    const float sx = par[PARAM_SX];
    const float sy = par[PARAM_SY];
    const float sz = par[PARAM_SZ];
    const float cutoff_mm = par[PARAM_CUTOFF];

    if (weq_acc >= cutoff_mm) {
        return;
    }

    const float eps_mm = 1.0e-4f * std::min(sx, std::min(sy, sz));
    float s_mm = s_start_mm + eps_mm;
    int i = static_cast<int>(floorf(ix + (s_mm * point.x) / sx + 0.5f));
    int j = static_cast<int>(floorf(iy + (s_mm * point.y) / sy + 0.5f));
    int k = static_cast<int>(floorf(iz + (s_mm * point.z) / sz + 0.5f));

    if (i < 0 || i >= resx || j < 0 || j >= resy || k < 0 || k >= nzs) {
        return;
    }

    const int stepx = point.x > 1.0e-6f ? 1 : (point.x < -1.0e-6f ? -1 : 0);
    const int stepy = point.y > 1.0e-6f ? 1 : (point.y < -1.0e-6f ? -1 : 0);
    const int stepz = point.z > 1.0e-6f ? 1 : (point.z < -1.0e-6f ? -1 : 0);

    const float x0 = ix + (s_mm * point.x) / sx;
    const float y0 = iy + (s_mm * point.y) / sy;
    const float z0 = iz + (s_mm * point.z) / sz;

    float tmax_x = LARGE_DISTANCE;
    float tmax_y = LARGE_DISTANCE;
    float tmax_z = LARGE_DISTANCE;
    float tdelta_x = LARGE_DISTANCE;
    float tdelta_y = LARGE_DISTANCE;
    float tdelta_z = LARGE_DISTANCE;

    if (stepx != 0) {
        const float boundary_x = static_cast<float>(i) + 0.5f * static_cast<float>(stepx);
        tmax_x = s_mm + (boundary_x - x0) * sx / point.x;
        tdelta_x = sx / fabsf(point.x);
    }
    if (stepy != 0) {
        const float boundary_y = static_cast<float>(j) + 0.5f * static_cast<float>(stepy);
        tmax_y = s_mm + (boundary_y - y0) * sy / point.y;
        tdelta_y = sy / fabsf(point.y);
    }
    if (stepz != 0) {
        const float boundary_z = static_cast<float>(k) + 0.5f * static_cast<float>(stepz);
        tmax_z = s_mm + (boundary_z - z0) * sz / point.z;
        tdelta_z = sz / fabsf(point.z);
    }

    while (i >= 0 && i < resx && j >= 0 && j < resy && k >= 0 && k < nzs) {
        float s_next = std::min(tmax_x, std::min(tmax_y, tmax_z));
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

        const float rho =
            std::max(sample_density_linear(density, resx, resy, nzs, i + 0.5f, j + 0.5f, k + 0.5f), MIN_DENSITY);
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
        const float weight = mixed_pdf_host(weq_mid, par) * used_dw;
        if (weight > 0.0f) {
            const int target_idx = voxel_index(i, j, k, resx, resy);
            (*weights)[target_idx] += weight;
            *total_weight += weight;
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

inline void build_row_for_source(const int source_idx,
                                 const std::vector<float>& density,
                                 const std::vector<float>& par,
                                 const std::vector<float3>& points,
                                 const int resx,
                                 const int resy,
                                 const int nzs,
                                 const float entry_eps,
                                 std::vector<int>* target_indices,
                                 std::vector<float>* row_weights) {
    const int plane = resx * resy;
    const int iz = source_idx / plane;
    const int rem = source_idx - iz * plane;
    const int iy = rem / resx;
    const int ix = rem - iy * resx;

    const float rho_source =
        std::max(sample_density_linear(density, resx, resy, nzs, ix + 0.5f, iy + 0.5f, iz + 0.5f), MIN_DENSITY);
    const int active_radius = near_field_radius_for_density_host(rho_source);

    std::unordered_map<int, float> weights;
    float total_weight = 0.0f;

    const float source_probability = local_voxel_probability_host(density, par, ix, iy, iz, 0, 0, 0, resx, resy, nzs);
    if (source_probability > 0.0f) {
        const float source_weight = static_cast<float>(points.size()) * source_probability;
        weights[source_idx] += source_weight;
        total_weight += source_weight;
    }

    for (int idz = -NEAR_FIELD_RADIUS; idz <= NEAR_FIELD_RADIUS; ++idz) {
        for (int idy = -NEAR_FIELD_RADIUS; idy <= NEAR_FIELD_RADIUS; ++idy) {
            for (int idx = -NEAR_FIELD_RADIUS; idx <= NEAR_FIELD_RADIUS; ++idx) {
                if ((idx == 0 && idy == 0 && idz == 0) || !inside_near_field_host(idx, idy, idz, active_radius)) {
                    continue;
                }
                const int tx = ix + idx;
                const int ty = iy + idy;
                const int tz = iz + idz;
                if (tx < 0 || tx >= resx || ty < 0 || ty >= resy || tz < 0 || tz >= nzs) {
                    continue;
                }
                const float prob =
                    local_voxel_probability_host(density, par, ix, iy, iz, idx, idy, idz, resx, resy, nzs);
                if (prob > 0.0f) {
                    const float near_weight = static_cast<float>(points.size()) * prob;
                    weights[voxel_index(tx, ty, tz, resx, resy)] += near_weight;
                    total_weight += near_weight;
                }
            }
        }
    }

    const float sx = par[PARAM_SX];
    const float sy = par[PARAM_SY];
    const float sz = par[PARAM_SZ];
    const float step_mm = par[PARAM_STEP];

    for (const float3 point : points) {
        const float near_exit_mm = ray_box_exit_mm_host(point, sx, sy, sz, static_cast<float>(active_radius) + 0.5f);
        const float near_weq_exit =
            integrate_weq_to_distance_host(density, point, ix, iy, iz, near_exit_mm, sx, sy, sz, step_mm, resx, resy,
                                           nzs);
        add_ray_weights_host(density, par, point, ix, iy, iz, resx, resy, nzs, near_exit_mm, near_weq_exit, &weights,
                             &total_weight);
    }

    if (!(total_weight > 0.0f)) {
        target_indices->clear();
        row_weights->clear();
        return;
    }

    float kept_normalized_sum = 0.0f;
    int best_index = -1;
    float best_normalized_weight = -1.0f;
    target_indices->reserve(weights.size());
    row_weights->reserve(weights.size());

    for (const auto& entry : weights) {
        const float normalized_weight = entry.second / total_weight;
        if (normalized_weight > best_normalized_weight) {
            best_normalized_weight = normalized_weight;
            best_index = entry.first;
        }
        if (normalized_weight > entry_eps) {
            target_indices->push_back(entry.first);
            row_weights->push_back(normalized_weight);
            kept_normalized_sum += normalized_weight;
        }
    }

    if (target_indices->empty() && best_index >= 0) {
        target_indices->push_back(best_index);
        row_weights->push_back(1.0f);
        return;
    }

    if (!(kept_normalized_sum > 0.0f)) {
        target_indices->clear();
        row_weights->clear();
        return;
    }

    for (float& weight : *row_weights) {
        weight /= kept_normalized_sum;
    }
}

inline std::vector<int> build_active_source_list(std::vector<float>* estimate, const float support_eps) {
    std::vector<int> active_sources;
    active_sources.reserve(estimate->size());

    for (size_t idx = 0; idx < estimate->size(); ++idx) {
        if ((*estimate)[idx] > support_eps) {
            active_sources.push_back(static_cast<int>(idx));
        } else {
            (*estimate)[idx] = 0.0f;
        }
    }

    return active_sources;
}

inline void apply_forward_tile(const TileCache& tile,
                               const std::vector<float>& estimate,
                               std::vector<float>* estimated) {
    for (size_t row = 0; row < tile.source_ids.size(); ++row) {
        const float source_value = estimate[tile.source_ids[row]];
        if (!(source_value > 0.0f)) {
            continue;
        }
        const int start = tile.row_offsets[row];
        const int end = tile.row_offsets[row + 1];
        for (int ptr = start; ptr < end; ++ptr) {
            (*estimated)[tile.target_indices[ptr]] += source_value * tile.weights[ptr];
        }
    }
}

inline void apply_backward_tile(const TileCache& tile,
                                const std::vector<float>& ratio,
                                std::vector<float>* estimate) {
    for (size_t row = 0; row < tile.source_ids.size(); ++row) {
        float dot = 0.0f;
        const int start = tile.row_offsets[row];
        const int end = tile.row_offsets[row + 1];
        for (int ptr = start; ptr < end; ++ptr) {
            dot += tile.weights[ptr] * ratio[tile.target_indices[ptr]];
        }
        (*estimate)[tile.source_ids[row]] *= dot;
        if ((*estimate)[tile.source_ids[row]] < 0.0f) {
            (*estimate)[tile.source_ids[row]] = 0.0f;
        }
    }
}

inline void compute_tv_denominator_host(const std::vector<float>& image,
                                        std::vector<float>* tv_denom,
                                        const int resx,
                                        const int resy,
                                        const int nzs,
                                        const float tv_weight,
                                        const float tv_eps) {
    const int nvoxels = resx * resy * nzs;
    std::vector<float> px(nvoxels, 0.0f);
    std::vector<float> py(nvoxels, 0.0f);
    std::vector<float> pz(nvoxels, 0.0f);

    for (int z = 0; z < nzs; ++z) {
        for (int y = 0; y < resy; ++y) {
            for (int x = 0; x < resx; ++x) {
                const int idx = voxel_index(x, y, z, resx, resy);
                const float u = image[idx];
                const float gx = x + 1 < resx ? image[voxel_index(x + 1, y, z, resx, resy)] - u : 0.0f;
                const float gy = y + 1 < resy ? image[voxel_index(x, y + 1, z, resx, resy)] - u : 0.0f;
                const float gz = z + 1 < nzs ? image[voxel_index(x, y, z + 1, resx, resy)] - u : 0.0f;
                const float mag = sqrtf(gx * gx + gy * gy + gz * gz + tv_eps * tv_eps);
                px[idx] = gx / mag;
                py[idx] = gy / mag;
                pz[idx] = gz / mag;
            }
        }
    }

    for (int z = 0; z < nzs; ++z) {
        for (int y = 0; y < resy; ++y) {
            for (int x = 0; x < resx; ++x) {
                const int idx = voxel_index(x, y, z, resx, resy);
                const float divx = px[idx] - (x > 0 ? px[voxel_index(x - 1, y, z, resx, resy)] : 0.0f);
                const float divy = py[idx] - (y > 0 ? py[voxel_index(x, y - 1, z, resx, resy)] : 0.0f);
                const float divz = pz[idx] - (z > 0 ? pz[voxel_index(x, y, z - 1, resx, resy)] : 0.0f);
                (*tv_denom)[idx] = std::max(1.0f - tv_weight * (divx + divy + divz), 0.5f);
            }
        }
    }
}

inline void apply_backward_tile_joint_tv(const TileCache& tile,
                                          const std::vector<float>& ratio,
                                          const std::vector<float>& tv_denom,
                                          std::vector<float>* estimate) {
    for (size_t row = 0; row < tile.source_ids.size(); ++row) {
        float dot = 0.0f;
        const int start = tile.row_offsets[row];
        const int end = tile.row_offsets[row + 1];
        for (int ptr = start; ptr < end; ++ptr) {
            dot += tile.weights[ptr] * ratio[tile.target_indices[ptr]];
        }
        const int src = tile.source_ids[row];
        const float updated = (*estimate)[src] * dot / tv_denom[src];
        (*estimate)[src] = updated > 0.0f ? updated : 0.0f;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 12 || argc > 17) {
        std::cerr << "Usage: " << argv[0]
                  << " ISOTOPE RESX RESY NZS DX DY DZ NRAYS ITERATIONS TV_WEIGHT TILE_SOURCES [TV_EPS] [EPSILON] [VERBOSE] [SUPPORT_EPS] [ENTRY_EPS]\n";
        std::cerr << "Reads blurred input from ACTIVITY.raw and density from DENSITY.raw.\n";
        std::cerr << "This version keeps only active source rows and nonzero coefficients of A in host memory.\n";
        print_supported_isotopes();
        return EXIT_FAILURE;
    }

    const std::string isotope_name = argv[1];
    const IsotopeModel* isotope_ref = find_isotope_model(isotope_name);
    if (isotope_ref == nullptr) {
        std::cerr << "Unsupported isotope: " << isotope_name << '\n';
        print_supported_isotopes();
        return EXIT_FAILURE;
    }

    IsotopeModel isotope = *isotope_ref;
    normalize_branch_weights(&isotope);

    const int RESX = std::atoi(argv[2]);
    const int RESY = std::atoi(argv[3]);
    const int NZS = std::atoi(argv[4]);
    const float dx = std::atof(argv[5]);
    const float dy = std::atof(argv[6]);
    const float dz = std::atof(argv[7]);
    const int N_RAYS = std::atoi(argv[8]);
    const int iterations = std::atoi(argv[9]);
    const float tv_weight = std::atof(argv[10]);
    const int tile_sources = std::atoi(argv[11]);
    const float tv_eps = argc >= 13 ? std::atof(argv[12]) : 1.0e-3f;
    const float epsilon = argc >= 14 ? std::atof(argv[13]) : 1.0e-6f;
    const bool verbose = argc >= 15 ? std::atoi(argv[14]) != 0 : false;
    const float support_eps = argc >= 16 ? std::atof(argv[15]) : 0.0f;
    const float entry_eps = argc >= 17 ? std::atof(argv[16]) : 0.0f;

    if (RESX <= 0 || RESY <= 0 || NZS <= 0 || dx <= 0.0f || dy <= 0.0f || dz <= 0.0f || N_RAYS <= 0 ||
        iterations <= 0 || tile_sources <= 0 || tv_eps <= 0.0f || epsilon <= 0.0f || support_eps < 0.0f ||
        entry_eps < 0.0f || entry_eps >= 1.0f) {
        std::cerr << "Invalid arguments.\n";
        return EXIT_FAILURE;
    }

    const size_t NVOXELS = voxel_count(RESX, RESY, NZS);
    std::vector<float> measured(NVOXELS, 0.0f);
    std::vector<float> density(NVOXELS, 0.0f);
    std::vector<float> estimate(NVOXELS, 0.0f);
    std::vector<float> estimated(NVOXELS, 0.0f);
    std::vector<float> ratio(NVOXELS, 0.0f);
    std::vector<float> reprojection(NVOXELS, 0.0f);
    std::vector<float> parameters(NPARAMS, 0.0f);
    std::vector<float3> points(N_RAYS);
    const auto total_start = std::chrono::steady_clock::now();

    if (!read_raw_file("ACTIVITY.raw", &measured) || !read_raw_file("DENSITY.raw", &density)) {
        return EXIT_FAILURE;
    }

    sanitize_input_images(&measured, &density);
    fill_fibonacci_points(&points);
    const float cutoff_mm = compute_cutoff_mm(isotope, CUTOFF_TARGET_CDF);
    fill_parameter_vector(isotope, dx, dy, dz, cutoff_mm, &parameters);
    estimate = measured;  // Initial estimate is the blurred image, not a uniform image.
    const std::vector<int> active_sources = build_active_source_list(&estimate, support_eps);

    const float observed_sum = sum_image(measured);
    const int total_sources = static_cast<int>(NVOXELS);
    const int active_source_count = static_cast<int>(active_sources.size());
    if (active_source_count == 0) {
        std::cerr << "No active source voxels remain after applying SUPPORT_EPS = " << support_eps << ".\n";
        return EXIT_FAILURE;
    }
    const int tile_count = (active_source_count + tile_sources - 1) / tile_sources;

    long long total_nnz = 0;
    size_t total_cache_bytes = 0;
    std::vector<TileCache> tiles;
    tiles.reserve(tile_count);
    if (entry_eps > 0.0f) {
        std::cerr << "WARNING: entry_eps = " << entry_eps
                  << " > 0 drops small matrix entries, narrowing the effective PSF. "
                  << "This accelerates noise amplification in RL iterations. "
                  << "Use entry_eps = 0 for best stability.\n";
    }

    std::cout << "Cached heterogeneous RL-TV: isotope = " << isotope_name << ", nrays = " << N_RAYS
              << ", iterations = " << iterations
              << ", tv_weight = " << tv_weight
              << ", tile_sources = " << tile_sources
              << ", support_eps = " << support_eps
              << ", entry_eps = " << entry_eps
              << ", active_sources = " << active_source_count << "/" << total_sources
              << ", cache = sparse host-memory rows"
              << ", init = blurred input" << '\n';

    const auto cache_build_start = std::chrono::steady_clock::now();
    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const int source_begin = tile_idx * tile_sources;
        const int source_end = std::min(source_begin + tile_sources, active_source_count);
        TileCache tile;
        tile.source_ids.reserve(source_end - source_begin);
        tile.row_offsets.reserve(static_cast<size_t>(source_end - source_begin) + 1);
        tile.row_offsets.push_back(0);

        for (int source_pos = source_begin; source_pos < source_end; ++source_pos) {
            const int source_idx = active_sources[source_pos];
            std::vector<int> row_targets;
            std::vector<float> row_weights;
            build_row_for_source(source_idx, density, parameters, points, RESX, RESY, NZS, entry_eps, &row_targets,
                                 &row_weights);
            if (row_targets.empty()) {
                continue;
            }
            tile.source_ids.push_back(source_idx);
            tile.target_indices.insert(tile.target_indices.end(), row_targets.begin(), row_targets.end());
            tile.weights.insert(tile.weights.end(), row_weights.begin(), row_weights.end());
            tile.row_offsets.push_back(static_cast<int>(tile.target_indices.size()));
        }

        if (tile.source_ids.empty()) {
            continue;
        }
        total_nnz += static_cast<long long>(tile.target_indices.size());
        total_cache_bytes += tile.source_ids.size() * sizeof(int);
        total_cache_bytes += tile.row_offsets.size() * sizeof(int);
        total_cache_bytes += tile.target_indices.size() * sizeof(int);
        total_cache_bytes += tile.weights.size() * sizeof(float);
        tiles.push_back(std::move(tile));
        if (verbose) {
            std::cout << "cached tile " << (tile_idx + 1) << "/" << tile_count
                      << ", sources = " << (source_end - source_begin)
                      << ", nnz = " << tiles.back().target_indices.size() << '\n';
        }
    }
    const auto cache_build_end = std::chrono::steady_clock::now();
    const double cache_build_s = std::chrono::duration<double>(cache_build_end - cache_build_start).count();

    std::cout << "Cache build complete: tiles = " << tile_count
              << ", stored_rows = " << [&tiles]() {
                     size_t rows = 0;
                     for (const TileCache& tile : tiles) {
                         rows += tile.source_ids.size();
                     }
                     return rows;
                 }()
              << ", total_nnz = " << total_nnz
              << ", cache_mb = " << (static_cast<double>(total_cache_bytes) / (1024.0 * 1024.0))
              << '\n';

    std::vector<float> tv_denom(NVOXELS, 1.0f);
    const auto iteration_loop_start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(estimated.begin(), estimated.end(), 0.0f);

        for (const TileCache& tile : tiles) {
            apply_forward_tile(tile, estimate, &estimated);
        }

        for (size_t idx = 0; idx < NVOXELS; ++idx) {
            ratio[idx] = measured[idx] / std::max(estimated[idx], epsilon);
        }

        if (tv_weight > 0.0f) {
            compute_tv_denominator_host(estimate, &tv_denom, RESX, RESY, NZS, tv_weight, tv_eps);
            for (const TileCache& tile : tiles) {
                apply_backward_tile_joint_tv(tile, ratio, tv_denom, &estimate);
            }
        } else {
            for (const TileCache& tile : tiles) {
                apply_backward_tile(tile, ratio, &estimate);
            }
        }

        const float estimate_sum = sum_image(estimate);
        if (estimate_sum > 0.0f && observed_sum > 0.0f) {
            const float scale = observed_sum / estimate_sum;
            for (float& value : estimate) {
                value = std::max(value * scale, 0.0f);
            }
        }

        if (verbose) {
            float max_value = 0.0f;
            for (const float value : estimate) {
                max_value = std::max(max_value, value);
            }
            std::cout << "iter = " << (iter + 1) << ", sum = " << sum_image(estimate)
                      << ", max = " << max_value << '\n';
        }
    }
    const auto iteration_loop_end = std::chrono::steady_clock::now();
    const double iteration_loop_s = std::chrono::duration<double>(iteration_loop_end - iteration_loop_start).count();

    const auto reprojection_start = std::chrono::steady_clock::now();
    std::fill(reprojection.begin(), reprojection.end(), 0.0f);
    for (const TileCache& tile : tiles) {
        apply_forward_tile(tile, estimate, &reprojection);
    }
    const auto reprojection_end = std::chrono::steady_clock::now();
    const double reprojection_s = std::chrono::duration<double>(reprojection_end - reprojection_start).count();
    const double total_runtime_s = std::chrono::duration<double>(reprojection_end - total_start).count();

    std::printf("SUM_MEASURED = %f\n", observed_sum);
    std::printf("SUM_CORRECTED = %f\n", sum_image(estimate));
    std::printf("SUM_REPROJECTED = %f\n", sum_image(reprojection));
    std::printf("CACHE_BUILD_S = %.6f\n", cache_build_s);
    std::printf("ITERATION_LOOP_S = %.6f\n", iteration_loop_s);
    std::printf("TIME_PER_ITERATION_S = %.6f\n", iteration_loop_s / static_cast<double>(iterations));
    std::printf("REPROJECTION_S = %.6f\n", reprojection_s);
    std::printf("TOTAL_RUNTIME_S = %.6f\n", total_runtime_s);

    write_raw_file("ACTIVITY_PRC_RL_HETERO_CACHE.raw", estimate);
    write_raw_file("ACTIVITY_PRC_RL_HETERO_CACHE_REPROJ.raw", reprojection);

    return EXIT_SUCCESS;
}
