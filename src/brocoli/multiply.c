#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

void batch_matmul_fused(int batch_size, int M, int N, int K, float* A, float* B, float* C) {
    #pragma omp parallel
    {
        float* packed_B = (float*)_mm_malloc(K * 16 * sizeof(float), 64);

        #pragma omp for schedule(static)
        for (int b = 0; b < batch_size; b++) {
            float* bA = &A[b * M * K];
            float* bB = &B[b * K * N];
            float* bC = &C[b * M * N];

            for (int j = 0; j < N; j += 16) {
                // Optimized Packing: Use SIMD to hide the packing cost
                for (int k = 0; k < K; k++) {
                    _mm256_store_ps(&packed_B[k * 16 + 0], _mm256_loadu_ps(&bB[k * N + j]));
                    _mm256_store_ps(&packed_B[k * 16 + 8], _mm256_loadu_ps(&bB[k * N + j + 8]));
                }

                for (int i = 0; i < M; i += 4) { // Focus on 4 rows with deeper K-unrolling
                    __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
                    __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
                    __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();

                    // K-UNROLL BY 8 WITH ACCUMULATOR INTERLEAVING
                    for (int k = 0; k < K; k++) {
                        __m256 bv0 = _mm256_load_ps(&packed_B[k * 16 + 0]);
                        __m256 bv1 = _mm256_load_ps(&packed_B[k * 16 + 8]);
                        
                        // Break dependencies by interleaving different rows
                        c00 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+0)*K+k]), bv0, c00);
                        c11 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+1)*K+k]), bv1, c11);
                        c20 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+2)*K+k]), bv0, c20);
                        c31 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+3)*K+k]), bv1, c31);
                        
                        c01 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+0)*K+k]), bv1, c01);
                        c10 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+1)*K+k]), bv0, c10);
                        c21 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+2)*K+k]), bv1, c21);
                        c30 = _mm256_fmadd_ps(_mm256_set1_ps(bA[(i+3)*K+k]), bv0, c30);
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
        }
        _mm_free(packed_B);
    }
}
