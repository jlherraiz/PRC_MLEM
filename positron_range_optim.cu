#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "kernel.h"
#include "kernel_optim.cu"
#include "positron_range_host_common.hpp"

int main(int argc, char** argv) {
    if (argc != 8 && argc != 9) {
        std::cerr << "Usage: " << argv[0] << " ISOTOPE RESX RESY NZS DX DY DZ [NRAYS]\n";
        std::cerr << "If NRAYS is omitted, a heuristic recommendation is used.\n";
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
    const float cutoff_mm = compute_cutoff_mm(isotope, CUTOFF_TARGET_CDF);
    const int recommended_nrays = recommend_nrays(cutoff_mm, dx, dy, dz);
    const int recommended_np = recommend_np(cutoff_mm, dx, dy, dz);
    const bool using_recommended_nrays = argc != 9;
    const int N_RAYS = argc == 9 ? std::atoi(argv[8]) : recommended_nrays;

    if (RESX <= 0 || RESY <= 0 || NZS <= 0 || dx <= 0.0f || dy <= 0.0f || dz <= 0.0f || N_RAYS <= 0) {
        std::cerr << "Invalid geometry arguments.\n";
        return EXIT_FAILURE;
    }

    const size_t NVOXELS = voxel_count(RESX, RESY, NZS);
    std::vector<float> ACTIVITY(NVOXELS, 0.0f);
    std::vector<float> DENSITY(NVOXELS, 0.0f);
    std::vector<float> POSITRON(NVOXELS, 0.0f);
    std::vector<float> PARAMETERS(NPARAMS, 0.0f);
    std::vector<float3> POINTS(N_RAYS);

    if (!read_raw_file("ACTIVITY.raw", &ACTIVITY) || !read_raw_file("DENSITY.raw", &DENSITY)) {
        return EXIT_FAILURE;
    }

    sanitize_input_images(&ACTIVITY, &DENSITY);
    fill_fibonacci_points(&POINTS);
    fill_parameter_vector(isotope, dx, dy, dz, cutoff_mm, &PARAMETERS);

    std::cout << "Isotope = " << isotope_name << ", branches = " << isotope.branches
              << ", step_mm = " << PARAMETERS[HOST_PARAM_STEP]
              << ", cutoff_weq_mm = " << PARAMETERS[HOST_PARAM_CUTOFF]
              << ", nrays = " << N_RAYS
              << (using_recommended_nrays ? " (heuristic)" : " (user)")
              << ", recommended_nrays = " << recommended_nrays
              << ", recommended_np = " << recommended_np << '\n';

    float* d_ACTIVITY = nullptr;
    float* d_PARAMETERS = nullptr;
    float* d_POSITRON = nullptr;
    float3* d_POINTS = nullptr;

    cudaMalloc((void**)&d_ACTIVITY, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_PARAMETERS, NPARAMS * sizeof(float));
    cudaMalloc((void**)&d_POSITRON, NVOXELS * sizeof(float));
    cudaMalloc((void**)&d_POINTS, N_RAYS * sizeof(float3));

    cudaMemcpy(d_ACTIVITY, ACTIVITY.data(), NVOXELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PARAMETERS, PARAMETERS.data(), NPARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_POINTS, POINTS.data(), N_RAYS * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_POSITRON, 0, NVOXELS * sizeof(float));

    cudaTextureObject_t DENS_tex = create_texture_density(DENSITY.data(), RESX, RESY, NZS);

    POSITRONRANGE(DENS_tex, d_POSITRON, d_ACTIVITY, d_POINTS, d_PARAMETERS, RESX, RESY, NZS, N_RAYS);
    chequearError("POSITRONRANGE");

    cudaMemcpy(POSITRON.data(), d_POSITRON, NVOXELS * sizeof(float), cudaMemcpyDeviceToHost);

    const float sum_in = sum_image(ACTIVITY);
    const float sum_out = sum_image(POSITRON);
    std::printf("SUM_IN = %f\n", sum_in);
    std::printf("SUM_OUT = %f\n", sum_out);

    write_raw_file("ACTIVITY_PRC.raw", POSITRON);

    cudaDestroyTextureObject(DENS_tex);
    cudaFree(d_POINTS);
    cudaFree(d_ACTIVITY);
    cudaFree(d_PARAMETERS);
    cudaFree(d_POSITRON);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
