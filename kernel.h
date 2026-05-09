#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <cuda_runtime.h>

void POSITRONRANGE(cudaTextureObject_t DENS_tex,
                   float* d_POSITRON,
                   const float* d_PET,
                   const float3* d_POINTS,
                   const float* d_PARAMETERS,
                   int RESX,
                   int RESY,
                   int NZS,
                   int N);
void POSITRONRANGE_EX(cudaTextureObject_t DENS_tex,
                      float* d_POSITRON,
                      const float* d_PET,
                      const float3* d_POINTS,
                      const float* d_PARAMETERS,
                      int RESX,
                      int RESY,
                      int NZS,
                      int N,
                      bool verbose);

cudaTextureObject_t create_texture_density(const float* h_imagen, int RESX, int RESY, int NZS);
void chequearError(const char* errorStr);

#endif
