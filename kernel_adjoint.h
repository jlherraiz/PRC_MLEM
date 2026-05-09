#ifndef KERNEL_ADJOINT_CUH_
#define KERNEL_ADJOINT_CUH_

#include <cuda_runtime.h>

void POSITRONRANGEADJOINT(cudaTextureObject_t DENS_tex,
                          float* d_OUTPUT,
                          const float* d_INPUT,
                          const float3* d_POINTS,
                          const float* d_PARAMETERS,
                          int RESX,
                          int RESY,
                          int NZS,
                          int N);
void POSITRONRANGEADJOINT_EX(cudaTextureObject_t DENS_tex,
                             float* d_OUTPUT,
                             const float* d_INPUT,
                             const float3* d_POINTS,
                             const float* d_PARAMETERS,
                             int RESX,
                             int RESY,
                             int NZS,
                             int N,
                             bool verbose);

cudaTextureObject_t create_texture_density_adjoint(const float* h_imagen, int RESX, int RESY, int NZS);
void chequearErrorAdjoint(const char* errorStr);

#endif
