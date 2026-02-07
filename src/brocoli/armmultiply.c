#include <arm_neon.h>
#include <omp.h>
#include <stdlib.h>

void batch_matmul_fused(int batch_size, int M, int N, int K, float* A, float* B, float* C) {
    #pragma omp parallel
    {
        // ARM NEON registers are 128-bit; align to 16 bytes
        float* packed_B = (float*)aligned_alloc(16, K * 8 * sizeof(float));

        #pragma omp for schedule(static)
        for (int b = 0; b < batch_size; b++) {
            float* bA = &A[b * M * K];
            float* bB = &B[b * K * N];
            float* bC = &C[b * M * N];

            for (int j = 0; j < N; j += 8) {
                // Pack B panel into 8-column strips
                for (int k = 0; k < K; k++) {
                    vst1q_f32(&packed_B[k * 8 + 0], vld1q_f32(&bB[k * N + j]));
                    vst1q_f32(&packed_B[k * 8 + 4], vld1q_f32(&bB[k * N + j + 4]));
                }

                for (int i = 0; i < M; i += 4) {
                    // Initialize 8 NEON accumulators (4x2 grid)
                    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
                    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
                    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
                    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);

                    for (int k = 0; k < K; k++) {
                        float32x4_t bv0 = vld1q_f32(&packed_B[k * 8 + 0]);
                        float32x4_t bv1 = vld1q_f32(&packed_B[k * 8 + 4]);

                        // Fused Multiply-Add (FMA) on ARM
                        c00 = vfmaq_n_f32(c00, bv0, bA[(i+0)*K+k]);
                        c01 = vfmaq_n_f32(c01, bv1, bA[(i+0)*K+k]);
                        c10 = vfmaq_n_f32(c10, bv0, bA[(i+1)*K+k]);
                        c11 = vfmaq_n_f32(c11, bv1, bA[(i+1)*K+k]);
                        c20 = vfmaq_n_f32(c20, bv0, bA[(i+2)*K+k]);
                        c21 = vfmaq_n_f32(c21, bv1, bA[(i+2)*K+k]);
                        c30 = vfmaq_n_f32(c30, bv0, bA[(i+3)*K+k]);
                        c31 = vfmaq_n_f32(c31, bv1, bA[(i+3)*K+k]);
                    }

                    // ReLU Activation (Max with Zero)
                    float32x4_t v_zero = vdupq_n_f32(0);
                    vst1q_f32(&bC[(i+0)*N+j], vmaxq_f32(c00, v_zero));
                    vst1q_f32(&bC[(i+0)*N+j+4], vmaxq_f32(c01, v_zero));
                    vst1q_f32(&bC[(i+1)*N+j], vmaxq_f32(c10, v_zero));
                    vst1q_f32(&bC[(i+1)*N+j+4], vmaxq_f32(c11, v_zero));
                    vst1q_f32(&bC[(i+2)*N+j], vmaxq_f32(c20, v_zero));
                    vst1q_f32(&bC[(i+2)*N+j+4], vmaxq_f32(c21, v_zero));
                    vst1q_f32(&bC[(i+3)*N+j], vmaxq_f32(c30, v_zero));
                    vst1q_f32(&bC[(i+3)*N+j+4], vmaxq_f32(c31, v_zero));
                }
            }
        }
        free(packed_B);
    }
}