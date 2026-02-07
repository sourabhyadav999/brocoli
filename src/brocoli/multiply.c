#include <omp.h>
#include <stdlib.h>

// 1. ARCHITECTURE DETECTION
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define IS_X86 1
    #define ALIGNMENT 64
    #define MALLOC(size) _mm_malloc(size, ALIGNMENT)
    #define FREE(ptr) _mm_free(ptr)
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define IS_ARM 1
    #define ALIGNMENT 16
    #include <malloc.h>
    // Note: On Linux/ARM use aligned_alloc; on Windows ARM use _aligned_malloc
    #define MALLOC(size) malloc(size) 
    #define FREE(ptr) free(ptr)
#endif

void batch_matmul_fused(int batch_size, int M, int N, int K, float* A, float* B, float* C) {
    #pragma omp parallel
    {
        #if IS_X86
            float* packed_B = (float*)MALLOC(K * 16 * sizeof(float));
        #else
            float* packed_B = (float*)MALLOC(K * 8 * sizeof(float));
        #endif

        #pragma omp for schedule(static)
        for (int b = 0; b < batch_size; b++) {
            float* bA = &A[b * M * K];
            float* bB = &B[b * K * N];
            float* bC = &C[b * M * N];

            #if IS_X86
            // --- INTEL/AMD AVX2 PATH (8-wide SIMD) ---
            for (int j = 0; j < N; j += 16) {
                for (int k = 0; k < K; k++) {
                    _mm256_store_ps(&packed_B[k * 16 + 0], _mm256_loadu_ps(&bB[k * N + j]));
                    _mm256_store_ps(&packed_B[k * 16 + 8], _mm256_loadu_ps(&bB[k * N + j + 8]));
                }
                for (int i = 0; i < M; i += 4) {
                    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
                    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
                    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

                    for (int k = 0; k < K; k++) {
                        __m256 bv0 = _mm256_load_ps(&packed_B[k * 16 + 0]);
                        __m256 bv1 = _mm256_load_ps(&packed_B[k * 16 + 8]);
                        c00 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+0)*K+k]), bv0, c00);
                        c01 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+0)*K+k]), bv1, c01);
                        c10 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+1)*K+k]), bv0, c10);
                        c11 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+1)*K+k]), bv1, c11);
                        c20 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+2)*K+k]), bv0, c20);
                        c21 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+2)*K+k]), bv1, c21);
                        c30 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+3)*K+k]), bv0, c30);
                        c31 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+3)*K+k]), bv1, c31);
                    }
                    __m256 z = _mm256_setzero_ps();
                    _mm256_storeu_ps(&bC[(i+0)*N+j], _mm256_max_ps(c00, z));
                    _mm256_storeu_ps(&bC[(i+0)*N+j+8], _mm256_max_ps(c01, z));
                    _mm256_storeu_ps(&bC[(i+1)*N+j], _mm256_max_ps(c10, z));
                    _mm256_storeu_ps(&bC[(i+1)*N+j+8], _mm256_max_ps(c11, z));
                    _mm256_storeu_ps(&bC[(i+2)*N+j], _mm256_max_ps(c20, z));
                    _mm256_storeu_ps(&bC[(i+2)*N+j+8], _mm256_max_ps(c21, z));
                    _mm256_storeu_ps(&bC[(i+3)*N+j], _mm256_max_ps(c30, z));
                    _mm256_storeu_ps(&bC[(i+3)*N+j+8], _mm256_max_ps(c31, z));
                }
            }
            #elif IS_ARM
            // --- ARM NEON PATH (4-wide SIMD) ---
            for (int j = 0; j < N; j += 8) {
                for (int k = 0; k < K; k++) {
                    vst1q_f32(&packed_B[k * 8 + 0], vld1q_f32(&bB[k * N + j]));
                    vst1q_f32(&packed_B[k * 8 + 4], vld1q_f32(&bB[k * N + j + 4]));
                }
                for (int i = 0; i < M; i += 4) {
                    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
                    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
                    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
                    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);

                    for (int k = 0; k < K; k++) {
                        float32x4_t bv0 = vld1q_f32(&packed_B[k * 8 + 0]);
                        float32x4_t bv1 = vld1q_f32(&packed_B[k * 8 + 4]);
                        c00 = vfmaq_n_f32(c00, bv0, bA[(i+0)*K+k]);
                        c01 = vfmaq_n_f32(c01, bv1, bA[(i+0)*K+k]);
                        c10 = vfmaq_n_f32(c10, bv0, bA[(i+1)*K+k]);
                        c11 = vfmaq_n_f32(c11, bv1, bA[(i+1)*K+k]);
                        c20 = vfmaq_n_f32(c20, bv0, bA[(i+2)*K+k]);
                        c21 = vfmaq_n_f32(c21, bv1, bA[(i+2)*K+k]);
                        c30 = vfmaq_n_f32(c30, bv0, bA[(i+3)*K+k]);
                        c31 = vfmaq_n_f32(c31, bv1, bA[(i+3)*K+k]);
                    }
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
            #endif
        }
        FREE(packed_B);
    }
}
