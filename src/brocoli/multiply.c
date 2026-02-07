#include <immintrin.h>
#include <omp.h>

/**
 * Parallel Matrix Multiplication (C = A * B)
 * Optimized with AVX2, FMA, and OpenMP.
 * * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Result Matrix C (M x N) - must be pre-zeroed
 */
void parallel_matmul_avx(int M, int N, int K, float* A, float* B, float* C) {
    // Parallelize the outer loop across CPU cores
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= M - 4; i += 4) {
        for (int j = 0; j <= N - 8; j += 8) {
            
            // Initialize 4 accumulators in registers
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                // Load B row once, reuse for 4 A rows
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                
                // Load/Broadcast A values and perform Fused Multiply-Add
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 0) * K + k]), b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 1) * K + k]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 2) * K + k]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 3) * K + k]), b_vec, c3);
            }

            // Store final sums back to memory ONLY ONCE
            _mm256_storeu_ps(&C[(i + 0) * N + j], c0);
            _mm256_storeu_ps(&C[(i + 1) * N + j], c1);
            _mm256_storeu_ps(&C[(i + 2) * N + j], c2);
            _mm256_storeu_ps(&C[(i + 3) * N + j], c3);
        }
    }
}
