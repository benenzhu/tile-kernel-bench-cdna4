#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(128) gemm_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  auto __rsrc_A = make_wave_buffer_resource((const void*)(A));
  uint32_t __base_A = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)(A));
  auto __rsrc_B = make_wave_buffer_resource((const void*)(B));
  uint32_t __base_B = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)(B));
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  bfloat16_t C_local_cast[4];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_lds_with_rsrc<16>((&(((bfloat16_t*)buf_dyn_shmem)[((i_1 * 1024) + (((int)threadIdx.x) * 8))])), (&(A[(((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __rsrc_A, __base_A);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    for (int vec_s = 0; vec_s < 8; ++vec_s) {
      tl::cp_async_gs_lds_with_rsrc<2>((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + vec_s) + 12288)])), (&(B[(((((((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 15) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + vec_s) - ((((int)threadIdx.x) & 7) * 8))])), __rsrc_B, __base_B);
    }
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs_lds_with_rsrc<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((i_3 * 1024) + (((int)threadIdx.x) * 8)) + 4096)])), (&(A[((((((((int)blockIdx.y) * 131072) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32)])), __rsrc_A, __base_A);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    for (int vec_s_1 = 0; vec_s_1 < 8; ++vec_s_1) {
      tl::cp_async_gs_lds_with_rsrc<2>((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + vec_s_1) + 16384)])), (&(B[((((((((((i_4 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 15) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + vec_s_1) + 32768) - ((((int)threadIdx.x) & 7) * 8))])), __rsrc_B, __base_B);
    }
  }
  tl::cp_async_commit();
  for (int k = 0; k < 30; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs_lds_with_rsrc<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((k + 2) % 3) * 4096) + (i_5 * 1024)) + (((int)threadIdx.x) * 8))])), (&(A[(((((((((int)blockIdx.y) * 131072) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)])), __rsrc_A, __base_A);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs_lds_with_rsrc<16>((&(((bfloat16_t*)buf_dyn_shmem)[((((((((k + 2) % 3) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 12288)])), (&(B[((((((((((k * 32768) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 15) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 65536) - ((((int)threadIdx.x) & 7) * 8))])), __rsrc_B, __base_B);
    }
    tl::cp_async_commit();
    tl::cp_async_wait<16>();
    __syncthreads();
    {
      bfloat16_t A_local[32];
      bfloat16_t B_local[64];
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        *(uint4*)(A_local + (i_7 * 8)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((k % 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)));
      }
      for (int j = 0; j < 8; ++j) {
        for (int local_id = 0; local_id < 8; ++local_id) {
          B_local[((j * 8) + local_id)] = ((bfloat16_t*)buf_dyn_shmem)[((((((((((k % 3) * 4096) + ((j >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 4) * 512)) + (local_id * 64)) + ((((local_id >> 2) + ((j & 3) >> 1)) & 1) * 32)) + (((((local_id & 3) >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 12288)];
        }
      }
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
      *(((float32x4*)C_local) + ((i_8 * 8) + j_1)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local) + j_1),
                    *(((bfloat16x8_vec*)A_local) + i_8),
                    *(((float32x4*)C_local) + ((i_8 * 8) + j_1)), 0, 0, 0);
    };
        }
      }
    }
  }
  tl::cp_async_wait<8>();
  __syncthreads();
  {
    bfloat16_t A_local_1[32];
    bfloat16_t B_local_1[64];
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      *(uint4*)(A_local_1 + (i_9 * 8)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 6) * 2048) + (i_9 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)));
    }
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      for (int local_id_1 = 0; local_id_1 < 8; ++local_id_1) {
        B_local_1[((j_2 * 8) + local_id_1)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((j_2 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 4) * 512)) + (local_id_1 * 64)) + ((((local_id_1 >> 2) + ((j_2 & 3) >> 1)) & 1) * 32)) + (((((local_id_1 & 3) >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_1 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 12288)];
      }
    }
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        {
      *(((float32x4*)C_local) + ((i_10 * 8) + j_3)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local_1) + j_3),
                    *(((bfloat16x8_vec*)A_local_1) + i_10),
                    *(((float32x4*)C_local) + ((i_10 * 8) + j_3)), 0, 0, 0);
    };
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    bfloat16_t A_local_2[32];
    bfloat16_t B_local_2[64];
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      *(uint4*)(A_local_2 + (i_11 * 8)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 6) * 2048) + (i_11 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096));
    }
    for (int j_4 = 0; j_4 < 8; ++j_4) {
      for (int local_id_2 = 0; local_id_2 < 8; ++local_id_2) {
        B_local_2[((j_4 * 8) + local_id_2)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((j_4 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 4) * 512)) + (local_id_2 * 64)) + ((((local_id_2 >> 2) + ((j_4 & 3) >> 1)) & 1) * 32)) + (((((local_id_2 & 3) >> 1) + (j_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_2 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 16384)];
      }
    }
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      for (int j_5 = 0; j_5 < 8; ++j_5) {
        {
      *(((float32x4*)C_local) + ((i_12 * 8) + j_5)) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(*(((bfloat16x8_vec*)B_local_2) + j_5),
                    *(((bfloat16x8_vec*)A_local_2) + i_12),
                    *(((float32x4*)C_local) + ((i_12 * 8) + j_5)), 0, 0, 0);
    };
      }
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 32; ++i_13) {
    uint2 __1;
    float4 v_ = *(float4*)(C_local + (i_13 * 4));
    ((bfloat16_t*)(&(__1.x)))[0] = (bfloat16_t)(v_.x);
    ((bfloat16_t*)(&(__1.x)))[1] = (bfloat16_t)(v_.y);
    ((bfloat16_t*)(&(__1.y)))[0] = (bfloat16_t)(v_.z);
    ((bfloat16_t*)(&(__1.y)))[1] = (bfloat16_t)(v_.w);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + (((((((((int)blockIdx.y) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + ((i_13 >> 3) * 16384)) + ((((int)threadIdx.x) & 15) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_13 & 7) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(uint2*)(C_local_cast + 0);
  }
}

