#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int MAX_BRANCHES = 3;
constexpr int BRANCH_STRIDE = 5;
constexpr int NPARAMS = 6 + MAX_BRANCHES * BRANCH_STRIDE;

constexpr int HOST_PARAM_SX = 0;
constexpr int HOST_PARAM_SY = 1;
constexpr int HOST_PARAM_SZ = 2;
constexpr int HOST_PARAM_STEP = 3;
constexpr int HOST_PARAM_CUTOFF = 4;
constexpr int HOST_PARAM_BRANCHES = 5;
constexpr int HOST_PARAM_BRANCH_BASE = 6;
constexpr float HEURISTIC_PI = 3.14159265358979323846f;
constexpr float CUTOFF_TARGET_CDF = 0.9995f;
constexpr float HOST_MIN_DENSITY = 1.0e-3f;

struct BranchModel {
    float weight;
    float a;
    float b;
    float c;
    float f;
};

struct IsotopeModel {
    const char* name;
    int branches;
    BranchModel branch[MAX_BRANCHES];
};

const IsotopeModel kIsotopeModels[] = {
    {"11C", 1, {{1.000f, 0.777f, 0.468f, 0.217f, 0.010f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    {"13N", 1, {{1.000f, 0.537f, 0.347f, 0.193f, 0.050f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    {"15O", 1, {{1.000f, 0.295f, 0.221f, 0.133f, 0.118f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    {"18F", 1, {{1.000f, 1.612f, 0.703f, 0.526f, 0.011f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    {"64Cu", 1, {{1.000f, 1.494f, 0.000f, 0.631f, 0.251f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    {"68Ga", 2, {{0.014f, 1.048f, 0.000f, 0.475f, 0.271f},
                  {0.986f, 0.289f, 0.137f, 0.159f, 0.322f},
                  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    {"82Rb", 2, {{0.136f, 0.191f, 0.084f, 0.114f, 0.408f},
                  {0.857f, 0.135f, 0.063f, 0.087f, 0.451f},
                  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
    {"124I", 3, {{0.013f, 1.007f, 0.000f, 0.421f, 0.443f},
                  {0.515f, 0.412f, 0.000f, 0.210f, 0.458f},
                  {0.471f, 0.241f, 0.000f, 0.150f, 0.369f}}},
};

inline float model_p(const BranchModel& branch, const float r_mm) {
    const float ar = branch.a * r_mm;
    const float br = branch.b * r_mm;
    const float cr = branch.c * r_mm;

    float p = std::pow(ar, 1.0f + branch.f);
    if (branch.b > 0.0f) {
        p += std::pow(br, 2.5f);
    }
    if (branch.c > 0.0f) {
        p += std::pow(cr, 3.5f);
    }
    return p;
}

inline float isotope_cdf(const IsotopeModel& isotope, const float r_mm) {
    float survival = 0.0f;
    for (int branch = 0; branch < isotope.branches; ++branch) {
        survival += isotope.branch[branch].weight * std::exp(-model_p(isotope.branch[branch], r_mm));
    }
    return 1.0f - survival;
}

inline float compute_cutoff_mm(const IsotopeModel& isotope, const float target_cdf) {
    float lo = 0.0f;
    float hi = 1.0f;
    while (isotope_cdf(isotope, hi) < target_cdf && hi < 256.0f) {
        hi *= 2.0f;
    }
    for (int iter = 0; iter < 64; ++iter) {
        const float mid = 0.5f * (lo + hi);
        if (isotope_cdf(isotope, mid) < target_cdf) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return hi;
}

inline int round_up_power_of_two(int value) {
    if (value <= 1) {
        return 1;
    }
    int power = 1;
    while (power < value && power < 16384) {
        power <<= 1;
    }
    return power;
}

inline int recommend_nrays(const float cutoff_mm, const float dx, const float dy, const float dz) {
    const float min_spacing = std::fmin(dx, std::fmin(dy, dz));
    const float ratio = cutoff_mm / min_spacing;
    const float nominal = 0.5f * 4.0f * HEURISTIC_PI * ratio * ratio;
    const int rounded = round_up_power_of_two(static_cast<int>(std::ceil(nominal)));
    return std::max(256, std::min(8192, rounded));
}

inline int recommend_np(const float cutoff_mm, const float dx, const float dy, const float dz) {
    const float min_spacing = std::fmin(dx, std::fmin(dy, dz));
    return std::max(1, static_cast<int>(std::ceil(cutoff_mm / min_spacing)));
}

inline float recommend_step_mm(const float dx, const float dy, const float dz) {
    return std::fmax(0.1f, 0.5f * std::fmin(dx, std::fmin(dy, dz)));
}

inline void normalize_branch_weights(IsotopeModel* isotope) {
    float sum = 0.0f;
    int min_branch = 0;
    for (int branch = 0; branch < isotope->branches; ++branch) {
        sum += isotope->branch[branch].weight;
        if (isotope->branch[branch].weight < isotope->branch[min_branch].weight) {
            min_branch = branch;
        }
    }
    isotope->branch[min_branch].weight += 1.0f - sum;
}

inline const IsotopeModel* find_isotope_model(const std::string& name) {
    for (const auto& isotope : kIsotopeModels) {
        if (name == isotope.name) {
            return &isotope;
        }
    }
    return nullptr;
}

inline bool read_raw_file(const char* filename, std::vector<float>* data) {
    FILE* file = fopen(filename, "rb");
    if (file == nullptr) {
        std::perror(filename);
        return false;
    }

    const size_t read = fread(data->data(), sizeof(float), data->size(), file);
    fclose(file);

    if (read != data->size()) {
        std::fprintf(stderr, "Failed to read %s: expected %zu floats, got %zu\n", filename, data->size(), read);
        return false;
    }

    return true;
}

inline void write_raw_file(const char* filename, const std::vector<float>& data) {
    FILE* file = fopen(filename, "wb");
    if (file == nullptr) {
        std::perror(filename);
        std::exit(EXIT_FAILURE);
    }

    const size_t written = fwrite(data.data(), sizeof(float), data.size(), file);
    fclose(file);

    if (written != data.size()) {
        std::fprintf(stderr, "Failed to write %s: expected %zu floats, got %zu\n", filename, data.size(), written);
        std::exit(EXIT_FAILURE);
    }
}

inline void sanitize_input_images(std::vector<float>* activity, std::vector<float>* density) {
    for (float& value : *activity) {
        if (value < 0.0f) {
            value = 0.0f;
        }
    }
    for (float& value : *density) {
        if (value < HOST_MIN_DENSITY) {
            value = HOST_MIN_DENSITY;
        }
    }
}

inline size_t voxel_count(const int resx, const int resy, const int nzs) {
    return static_cast<size_t>(resx) * static_cast<size_t>(resy) * static_cast<size_t>(nzs);
}

inline void fill_parameter_vector(const IsotopeModel& isotope,
                                  const float dx,
                                  const float dy,
                                  const float dz,
                                  const float cutoff_mm,
                                  std::vector<float>* parameters) {
    std::fill(parameters->begin(), parameters->end(), 0.0f);
    (*parameters)[HOST_PARAM_SX] = dx;
    (*parameters)[HOST_PARAM_SY] = dy;
    (*parameters)[HOST_PARAM_SZ] = dz;
    (*parameters)[HOST_PARAM_STEP] = recommend_step_mm(dx, dy, dz);
    (*parameters)[HOST_PARAM_CUTOFF] = cutoff_mm;
    (*parameters)[HOST_PARAM_BRANCHES] = static_cast<float>(isotope.branches);

    for (int branch = 0; branch < isotope.branches; ++branch) {
        const int base = HOST_PARAM_BRANCH_BASE + branch * BRANCH_STRIDE;
        (*parameters)[base + 0] = isotope.branch[branch].weight;
        (*parameters)[base + 1] = isotope.branch[branch].a;
        (*parameters)[base + 2] = isotope.branch[branch].b;
        (*parameters)[base + 3] = isotope.branch[branch].c;
        (*parameters)[base + 4] = isotope.branch[branch].f;
    }
}

inline void fill_fibonacci_points(std::vector<float3>* points) {
    const double pi = 3.14159265358979323846;
    const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    const int num_points = static_cast<int>(points->size());

    if (num_points <= 0) {
        return;
    }
    if (num_points == 1) {
        (*points)[0] = make_float3(0.0f, 0.0f, 1.0f);
        return;
    }

    for (int k = 0; k < num_points; ++k) {
        const double z = 1.0 - (2.0 * k) / (num_points - 1);
        const double theta = 2.0 * pi * k / golden_ratio;
        const double r = std::sqrt(1.0 - z * z);
        (*points)[k] = make_float3(static_cast<float>(r * std::cos(theta)),
                                   static_cast<float>(r * std::sin(theta)),
                                   static_cast<float>(z));
    }
}

inline float sum_image(const std::vector<float>& image) {
    double sum = 0.0;
    for (const float value : image) {
        sum += value;
    }
    return static_cast<float>(sum);
}

inline void print_supported_isotopes() {
    std::cerr << "Supported isotopes:";
    for (const auto& isotope : kIsotopeModels) {
        std::cerr << ' ' << isotope.name;
    }
    std::cerr << '\n';
}

}  // namespace
