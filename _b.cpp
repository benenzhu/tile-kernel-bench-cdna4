#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(512) gemm_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  float C_local[128];
  __shared__ __align__(1024) bfloat16_t A_shared[32768];
  __shared__ __align__(1024) bfloat16_t B_shared[32768];
  bfloat16_t C_local_cast[4];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(A_shared[(((((i_1 * 4096) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 2097152) + (i_1 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(B_shared[(((((i_2 * 4096) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((int)blockIdx.x) * 2097152) + (i_2 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 127; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(A_shared[((((((((k + 1) & 1) * 16384) + (i_3 * 4096)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) * 2097152) + (i_3 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>((&(B_shared[((((((((k + 1) & 1) * 16384) + (i_4 * 4096)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((int)blockIdx.x) * 2097152) + (i_4 * 524288)) + ((((int)threadIdx.x) >> 3) * 8192)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      bfloat16_t A_local[32];
      bfloat16_t B_local[64];
      for (int ki = 0; ki < 2; ++ki) {
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          *(uint4*)(A_local + (i_5 * 8)) = *(uint4*)(A_shared + ((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 255) >> 6) * 4096)) + (i_5 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        }
        for (int j = 0; j < 8; ++j) {
          *(uint4*)(B_local + (j * 8)) = *(uint4*)(B_shared + ((((((((k & 1) * 16384) + ((((int)threadIdx.x) >> 8) * 8192)) + (j * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        }
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          for (int j_1 = 0; j_1 < 8; ++j_1) {
            {
      *(((float32x4*)C_local) + ((i_6 * 8) + j_1)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + j_1),
                    *(((bfloat16x8_vec*)A_local) + i_6),
                    *(((float32x4*)C_local) + ((i_6 * 8) + j_1)), 0, 0, 0);
    };
          }
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    bfloat16_t A_local_1[32];
    bfloat16_t B_local_1[64];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        *(uint4*)(A_local_1 + (i_7 * 8)) = *(uint4*)(A_shared + (((((((((((int)threadIdx.x) & 255) >> 6) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384));
      }
      for (int j_2 = 0; j_2 < 8; ++j_2) {
        *(uint4*)(B_local_1 + (j_2 * 8)) = *(uint4*)(B_shared + ((((((((((int)threadIdx.x) >> 8) * 8192) + (j_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384));
      }
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        for (int j_3 = 0; j_3 < 8; ++j_3) {
          {
      *(((float32x4*)C_local) + ((i_8 * 8) + j_3)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local_1) + j_3),
                    *(((bfloat16x8_vec*)A_local_1) + i_8),
                    *(((float32x4*)C_local) + ((i_8 * 8) + j_3)), 0, 0, 0);
    };
        }
      }
    }
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    uint2 __1;
    float4 v_ = *(float4*)(C_local + (i_9 * 4));
    ((bfloat16_t*)(&(__1.x)))[0] = (bfloat16_t)(v_.x);
    ((bfloat16_t*)(&(__1.x)))[1] = (bfloat16_t)(v_.y);
    ((bfloat16_t*)(&(__1.y)))[0] = (bfloat16_t)(v_.z);
    ((bfloat16_t*)(&(__1.y)))[1] = (bfloat16_t)(v_.w);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + ((((((((((int)blockIdx.y) * 2097152) + (((((int)threadIdx.x) & 255) >> 6) * 524288)) + ((i_9 >> 3) * 131072)) + ((((int)threadIdx.x) & 15) * 8192)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) >> 8) * 128)) + ((i_9 & 7) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(uint2*)(C_local_cast + 0);
  }
}


#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif

#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" TL_EXPORT const char* get_last_error() {
    return error_buf;
}

extern "C" TL_EXPORT int init() {
    error_buf[0] = '\0';
    
    int device_gemm_kernel = 0;
    hipError_t dev_res_gemm_kernel = hipGetDevice(&device_gemm_kernel);
    if (dev_res_gemm_kernel != hipSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to get HIP device for gemm_kernel: %s", hipGetErrorString(dev_res_gemm_kernel));
        return -1;
    }
    int max_smem_gemm_kernel = 0;
    hipError_t attr_res_gemm_kernel = hipDeviceGetAttribute(&max_smem_gemm_kernel, hipDeviceAttributeMaxSharedMemoryPerBlock, device_gemm_kernel);
    if (attr_res_gemm_kernel != hipSuccess || max_smem_gemm_kernel <= 0) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to query HIP max shared memory for gemm_kernel: %s", hipGetErrorString(attr_res_gemm_kernel));
        return -1;
    }
    if (131072 > max_smem_gemm_kernel) {
        snprintf(
            error_buf,
            ERROR_BUF_SIZE,
            "Requested dynamic shared memory %d exceeds device limit %d for gemm_kernel",
            131072,
            max_smem_gemm_kernel
        );
        return -1;
    }
    return 0;

    return 0;
}

extern "C" TL_EXPORT int call(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C, hipStream_t stream=hipStreamDefault) {
	gemm_kernel<<<dim3(32, 32, 1), dim3(512, 1, 1), 0, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("gemm_kernel");

	return 0;
}
