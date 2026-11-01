#include "kernels.h"

static size_t min_size(size_t a, size_t b) {
    return (a < b) ? a : b;
}
void gemm_tiled(const float *a, const float *b, float *c,
                size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 6 — Tiled GEMM
       Arguments/layout: same A[M,K] * B[K,N] -> C[M,N] as gemm_scalar;
       tile is the positive tile width. Divide the i, j, and reduction q
       ranges into blocks. Clamp the final block in each dimension to M, N,
       or K (for example with min(block_start+tile, dimension)). This keeps
       a smaller working set active and can improve data reuse.
       Pseudocode:
         for ii in blocks of tile over M:
           for jj in blocks of tile over N:
             for qq in blocks of tile over K:
               for i in ii .. min(ii+tile,M):
                 for j in jj .. min(jj+tile,N):
                   for q in qq .. min(qq+tile,K):
                     accumulate a[i*K+q]*b[q*N+j] into c[i*N+j]
       Initialize C to zero before these accumulation loops. Compare tiles
       4, 8, 16, and 32. */
    /* The placeholder does not read these inputs/options yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    /* (void)a;
    (void)b;
    (void)k;
    (void)tile; */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[i * n + j] = 0.0f;
        }
    }
    for (size_t ii = 0; ii < m/tile + 1; ++ii) {
        for (size_t jj = 0; jj < n/tile + 1; ++jj) {
            for (size_t qq = 0; qq < k/tile + 1; ++qq) {
                for (size_t i = ii * tile; i < min_size((ii + 1) * tile, m); ++i) {
                    for (size_t j = jj * tile; j < min_size((jj + 1) * tile, n); ++j) {
                        for (size_t q = qq * tile; q < min_size((qq + 1) * tile, k); ++q) {
                            c[i * n + j] += a[i * k + q] * b[q * n + j];
                        }
                    }
                }
            }
        }
    }
}
