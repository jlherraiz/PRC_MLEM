#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "kernel.h"
#include "kernel_adjoint.h"
#include "positron_range_host_common.hpp"

namespace {

__global__ void COMPUTE_RATIO_KERNEL(float* ratio,
                                     const float* measured,
                                     const float* estimated,
                                     const int nvoxels,
                                     const float epsilon) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvoxels) {
        return;
    }
    ratio[idx] = measured[idx] / fmaxf(estimated[idx], epsilon);
}

__global__ void RL_UPDATE_KERNEL(float* estimate,
                                 const float* correction,
                                 const float* ones_backproj,
                                 const int nvoxels,
                                 const float epsilon) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvoxels) {
        return;
    }
    const float denom = fmaxf(ones_backproj[idx], epsilon);
    const float updated = estimate[idx] * correction[idx] / denom;
    estimate[idx] = updated > 0.0f ? updated : 0.0f;
}

__global__ void RL_TV_JOINT_UPDATE_KERNEL(float* estimate,
                                           const float* correction,
                                           const float* ones_backproj,
                                           const float* grad_px,
                                           const float* grad_py,
                                           const float* grad_pz,
                                           const int resx,
                                           const int resy,
                                           const int nzs,
                                           const float tv_weight,
                                           const float epsilon) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nvoxels = resx * resy * nzs;
    if (idx >= nvoxels) {
        return;
    }

    const int plane = resx * resy;
    const int z = idx / plane;
    const int rem = idx - z * plane;
    const int y = rem / resx;
    const int x = rem - y * resx;

    const float divx = grad_px[idx] - (x > 0 ? grad_px[idx - 1] : 0.0f);
    const float divy = grad_py[idx] - (y > 0 ? grad_py[idx - resx] : 0.0f);
    const float divz = grad_pz[idx] - (z > 0 ? grad_pz[idx - plane] : 0.0f);
    const float div = divx + divy + divz;

    const float rl_denom = fmaxf(ones_backproj[idx], epsilon);
    const float tv_denom = fmaxf(1.0f - tv_weight * div, 0.5f);
    const float updated = estimate[idx] * correction[idx] / (rl_denom * tv_denom);
    estimate[idx] = updated > 0.0f ? updated : 0.0f;
}

__global__ void SCALE_AND_CLAMP_KERNEL(float* image, const int nvoxels, const float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvoxels) {
        return;
    }
    const float scaled = image[idx] * scale;
    image[idx] = scaled > 0.0f ? scaled : 0.0f;
}

__device__ float SAMPLE_IMAGE_CLAMP(const float* image,
                                    const int x,
                                    const int y,
                                    const int z,
                                    const int resx,
                                    const int resy,
                                    const int nzs) {
    if (x < 0 || x >= resx || y < 0 || y >= resy || z < 0 || z >= nzs) {
        return 0.0f;
    }
    return image[z * resx * resy + y * resx + x];
}

__global__ void TV_COMPUTE_NORMALIZED_GRADIENT_KERNEL(const float* image,
                                                       float* grad_px,
                                                       float* grad_py,
                                                       float* grad_pz,
                                                       const int resx,
                                                       const int resy,
                                                       const int nzs,
                                                       const float tv_eps) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nvoxels = resx * resy * nzs;
    if (idx >= nvoxels) {
        return;
    }

    const int plane = resx * resy;
    const int z = idx / plane;
    const int rem = idx - z * plane;
    const int y = rem / resx;
    const int x = rem - y * resx;

    const float u = image[idx];
    const float gx = x + 1 < resx ? SAMPLE_IMAGE_CLAMP(image, x + 1, y, z, resx, resy, nzs) - u : 0.0f;
    const float gy = y + 1 < resy ? SAMPLE_IMAGE_CLAMP(image, x, y + 1, z, resx, resy, nzs) - u : 0.0f;
    const float gz = z + 1 < nzs ? SAMPLE_IMAGE_CLAMP(image, x, y, z + 1, resx, resy, nzs) - u : 0.0f;
    const float mag = sqrtf(gx * gx + gy * gy + gz * gz + tv_eps * tv_eps);
    grad_px[idx] = gx / mag;
    grad_py[idx] = gy / mag;
    grad_pz[idx] = gz / mag;
}

inline int ceil_div(const int a, const int b) {
    return (a + b - 1) / b;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 11 || argc > 14) {
        std::cerr << "Usage: " << argv[0]
                  << " ISOTOPE RESX RESY NZS DX DY DZ NRAYS ITERATIONS TV_WEIGHT [TV_EPS] [EPSILON] [VERBOSE]\n";
        std::cerr << "Reads blurred input from ACTIVITY.raw and density from DENSITY.raw.\n";
        std::cerr << "Initial estimate is the blurred input image itself.\n";
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
    const float tv_eps = argc >= 12 ? std::atof(argv[11]) : 1.0e-3f;
    const float epsilon = argc >= 13 ? std::atof(argv[12]) : 1.0e-6f;
    const bool verbose = argc >= 14 ? std::atoi(argv[13]) != 0 : false;

    if (RESX <= 0 || RESY <= 0 || NZS <= 0 || dx <= 0.0f || dy <= 0.0f || dz <= 0.0f || N_RAYS <= 0 ||
        iterations <= 0 || tv_eps <= 0.0f || epsilon <= 0.0f) {
        std::cerr << "Invalid arguments.\n";
        return EXIT_FAILURE;
    }

    const size_t NVOXELS = voxel_count(RESX, RESY, NZS);
    const int nvoxels_int = static_cast<int>(NVOXELS);
    const float cutoff_mm = compute_cutoff_mm(isotope, CUTOFF_TARGET_CDF);
    const int recommended_nrays = recommend_nrays(cutoff_mm, dx, dy, dz);
    const int recommended_np = recommend_np(cutoff_mm, dx, dy, dz);

    std::vector<float> measured(NVOXELS, 0.0f);
    std::vector<float> density(NVOXELS, 0.0f);
    std::vector<float> corrected(NVOXELS, 0.0f);
    std::vector<float> reprojection(NVOXELS, 0.0f);
    std::vector<float> parameters(NPARAMS, 0.0f);
    std::vector<float3> points(N_RAYS);
    std::vector<float> ones_host(NVOXELS, 1.0f);
    const auto total_start = std::chrono::steady_clock::now();

    if (!read_raw_file("ACTIVITY.raw", &measured) || !read_raw_file("DENSITY.raw", &density)) {
        return EXIT_FAILURE;
    }

    sanitize_input_images(&measured, &density);
    fill_fibonacci_points(&points);
    fill_parameter_vector(isotope, dx, dy, dz, cutoff_mm, &parameters);

    const float observed_sum = sum_image(measured);

    std::cout << "Optimized heterogeneous RL-TV: isotope = " << isotope_name << ", branches = " << isotope.branches
              << ", step_mm = " << parameters[HOST_PARAM_STEP]
              << ", cutoff_weq_mm = " << parameters[HOST_PARAM_CUTOFF]
              << ", nrays = " << N_RAYS
              << ", recommended_nrays = " << recommended_nrays
              << ", recommended_np = " << recommended_np
              << ", iterations = " << iterations
              << ", tv_weight = " << tv_weight
              << ", tv_eps = " << tv_eps
              << ", epsilon = " << epsilon
              << ", init = blurred input" << '\n';

    float* d_measured = nullptr;
    float* d_estimate = nullptr;
    float* d_estimated = nullptr;
    float* d_ratio = nullptr;
    float* d_correction = nullptr;
    float* d_ones = nullptr;
    float* d_ones_backproj = nullptr;
    float* d_parameters = nullptr;
    float3* d_points = nullptr;
    float* d_tv_px = nullptr;
    float* d_tv_py = nullptr;
    float* d_tv_pz = nullptr;

    cudaMalloc((void**)&d_measured, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_estimate, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_estimated, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_ratio, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_correction, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_ones, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_ones_backproj, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_parameters, NPARAMS * sizeof(float));
    cudaMalloc((void**)&d_points, N_RAYS * sizeof(float3));
    cudaMalloc((void**)&d_tv_px, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_tv_py, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_tv_pz, NVOXELS * sizeof(float));

    cudaMemcpy(d_measured, measured.data(), NVOXELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_estimate, measured.data(), NVOXELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ones, ones_host.data(), NVOXELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parameters, parameters.data(), NPARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, points.data(), N_RAYS * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_estimated, 0, NVOXELS * sizeof(float));
    cudaMemset(d_ratio, 0, NVOXELS * sizeof(float));
    cudaMemset(d_correction, 0, NVOXELS * sizeof(float));
    cudaMemset(d_ones_backproj, 0, NVOXELS * sizeof(float));

    cudaTextureObject_t DENS_tex = create_texture_density(density.data(), RESX, RESY, NZS);

    const auto ones_backproj_start = std::chrono::steady_clock::now();
    POSITRONRANGEADJOINT_EX(DENS_tex, d_ones_backproj, d_ones, d_points, d_parameters, RESX, RESY, NZS, N_RAYS,
                            false);
    chequearErrorAdjoint("POSITRONRANGEADJOINT_EX(ones)");
    const auto ones_backproj_end = std::chrono::steady_clock::now();
    const double ones_backproj_s = std::chrono::duration<double>(ones_backproj_end - ones_backproj_start).count();

    const int threads = 256;
    const int blocks = ceil_div(nvoxels_int, threads);
    thrust::device_ptr<float> estimate_ptr(d_estimate);

    const auto iteration_loop_start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        cudaMemset(d_estimated, 0, NVOXELS * sizeof(float));
        POSITRONRANGE_EX(DENS_tex, d_estimated, d_estimate, d_points, d_parameters, RESX, RESY, NZS, N_RAYS, false);
        chequearError("POSITRONRANGE_EX(iter)");

        COMPUTE_RATIO_KERNEL<<<blocks, threads>>>(d_ratio, d_measured, d_estimated, nvoxels_int, epsilon);
        chequearError("COMPUTE_RATIO_KERNEL");

        cudaMemset(d_correction, 0, NVOXELS * sizeof(float));
        POSITRONRANGEADJOINT_EX(DENS_tex, d_correction, d_ratio, d_points, d_parameters, RESX, RESY, NZS, N_RAYS,
                                false);
        chequearErrorAdjoint("POSITRONRANGEADJOINT_EX(iter)");

        if (tv_weight > 0.0f) {
            TV_COMPUTE_NORMALIZED_GRADIENT_KERNEL<<<blocks, threads>>>(d_estimate, d_tv_px, d_tv_py, d_tv_pz,
                                                                        RESX, RESY, NZS, tv_eps);
            chequearError("TV_COMPUTE_NORMALIZED_GRADIENT_KERNEL");

            RL_TV_JOINT_UPDATE_KERNEL<<<blocks, threads>>>(d_estimate, d_correction, d_ones_backproj,
                                                            d_tv_px, d_tv_py, d_tv_pz,
                                                            RESX, RESY, NZS, tv_weight, epsilon);
            chequearError("RL_TV_JOINT_UPDATE_KERNEL");
        } else {
            RL_UPDATE_KERNEL<<<blocks, threads>>>(d_estimate, d_correction, d_ones_backproj, nvoxels_int, epsilon);
            chequearError("RL_UPDATE_KERNEL");
        }

        const float estimate_sum = thrust::reduce(estimate_ptr, estimate_ptr + nvoxels_int, 0.0f, thrust::plus<float>());
        if (estimate_sum > 0.0f && observed_sum > 0.0f) {
            const float scale = observed_sum / estimate_sum;
            SCALE_AND_CLAMP_KERNEL<<<blocks, threads>>>(d_estimate, nvoxels_int, scale);
            chequearError("SCALE_AND_CLAMP_KERNEL");
        }

        if (verbose) {
            const float max_value =
                thrust::reduce(estimate_ptr, estimate_ptr + nvoxels_int, 0.0f, thrust::maximum<float>());
            const float sum_value =
                thrust::reduce(estimate_ptr, estimate_ptr + nvoxels_int, 0.0f, thrust::plus<float>());
            std::cout << "iter = " << (iter + 1) << ", sum = " << sum_value << ", max = " << max_value << '\n';
        }
    }
    const auto iteration_loop_end = std::chrono::steady_clock::now();
    const double iteration_loop_s = std::chrono::duration<double>(iteration_loop_end - iteration_loop_start).count();

    const auto reprojection_start = std::chrono::steady_clock::now();
    cudaMemset(d_estimated, 0, NVOXELS * sizeof(float));
    POSITRONRANGE_EX(DENS_tex, d_estimated, d_estimate, d_points, d_parameters, RESX, RESY, NZS, N_RAYS, false);
    chequearError("POSITRONRANGE_EX(final)");
    const auto reprojection_end = std::chrono::steady_clock::now();
    const double reprojection_s = std::chrono::duration<double>(reprojection_end - reprojection_start).count();

    cudaMemcpy(corrected.data(), d_estimate, NVOXELS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(reprojection.data(), d_estimated, NVOXELS * sizeof(float), cudaMemcpyDeviceToHost);
    const double total_runtime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();

    std::printf("SUM_MEASURED = %f\n", observed_sum);
    std::printf("SUM_CORRECTED = %f\n", sum_image(corrected));
    std::printf("SUM_REPROJECTED = %f\n", sum_image(reprojection));
    std::printf("ONES_BACKPROJ_S = %.6f\n", ones_backproj_s);
    std::printf("ITERATION_LOOP_S = %.6f\n", iteration_loop_s);
    std::printf("TIME_PER_ITERATION_S = %.6f\n", iteration_loop_s / static_cast<double>(iterations));
    std::printf("REPROJECTION_S = %.6f\n", reprojection_s);
    std::printf("TOTAL_RUNTIME_S = %.6f\n", total_runtime_s);

    write_raw_file("ACTIVITY_PRC_RL_HETERO_OPT.raw", corrected);
    write_raw_file("ACTIVITY_PRC_RL_HETERO_OPT_REPROJ.raw", reprojection);

    cudaDestroyTextureObject(DENS_tex);
    cudaFree(d_measured);
    cudaFree(d_estimate);
    cudaFree(d_estimated);
    cudaFree(d_ratio);
    cudaFree(d_correction);
    cudaFree(d_ones);
    cudaFree(d_ones_backproj);
    cudaFree(d_parameters);
    cudaFree(d_points);
    cudaFree(d_tv_px);
    cudaFree(d_tv_py);
    cudaFree(d_tv_pz);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
