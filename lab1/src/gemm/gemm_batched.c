#include "kernels.h"
#include <omp.h>

static size_t min_size(size_t a, size_t b) {
    return (a < b) ? a : b;
}

void gemm_batched_scalar(const float *a, const float *b, float *c,
                         size_t batch, size_t m, size_t n, size_t k) {
    /* TODO 7 — Batched scalar GEMM
       a stores batch independent row-major matrices A[batch,M,K]; b is one
       shared row-major weight matrix B[K,N]; c stores C[batch,M,N]. For each
       batch item, compute A[batch_i] * B using the ordinary GEMM dot product.
       Pseudocode: for t in batch, i in M, j in N: sum=0; for q in K:
       sum += a[(t*M+i)*K+q]*b[q*N+j]; c[(t*M+i)*N+j]=sum. */
    /* The placeholder does not read these inputs/dimensions yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    /* (void)a;
    (void)b;
    (void)k; */
    for (size_t t = 0; t < batch; ++t) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t q = 0; q < k; ++q) {
                    sum += a[(t * m + i) * k + q] * b[q * n + j];
                }
                c[(t * m + i) * n + j] = sum;
            }
        }
    }
}

void gemm_batched_tiled(const float *a, const float *b, float *c,
                        size_t batch, size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 8 — Batched tiled GEMM
       Arguments/layout are the same as gemm_batched_scalar; tile controls
       block widths. Tile each per-item output/reduction matrix as in
       gemm_tiled, but reuse the same b[K,N] for every batch item.
       Pseudocode: for t in batch: zero C[t]; for ii,jj,qq tile blocks:
       iterate bounded i,j,q ranges and accumulate A[t,i,q]*B[q,j]. */
    /* The placeholder does not read these inputs/options yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    /* (void)a;
    (void)b;
    (void)k;
    (void)tile; */
    for (size_t t = 0; t < batch; ++t) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c[(t * m + i) * n + j] = 0.0f;
            }
        }
         for (size_t ii = 0; ii < m/tile + 1; ++ii) {
               for (size_t jj = 0; jj < n/tile + 1; ++jj) {
                  for (size_t qq = 0; qq < k/tile + 1; ++qq) {
                     for (size_t i = ii * tile; i < min_size((ii + 1) * tile, m); ++i) {
                           for (size_t j = jj * tile; j < min_size((jj + 1) * tile, n); ++j) {
                              for (size_t q = qq * tile; q < min_size((qq + 1) * tile, k); ++q) {
                                 c[(t * m + i) * n + j] += a[(t * m + i) * k + q] * b[q * n + j];
                              }
                           }
                     }
                  }
               }
         }
    }
}

void gemm_batched_omp_batch(const float *a, const float *b, float *c,
                            size_t batch, size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 9 — OpenMP over batch
       Inputs/outputs have the same layout as above. Batch items write to
       disjoint C[t] regions, so parallelize the outer batch loop with one
       OpenMP parallel-for; keep each item's tiled GEMM serial. The shared
       B matrix is read-only. Pseudocode: parallel for t in batch: compute
       the complete tiled C[t] = A[t] * B. */
    /* The placeholder does not read these inputs/options yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    /* (void)a;
    (void)b;
    (void)k;
    (void)tile; */
    #pragma omp parallel for
    for (size_t t = 0; t < batch; ++t) {
        gemm_batched_tiled(a + t * m * k, b, c + t * m * n, 1, m, n, k, tile);
    }
}

void gemm_batched_omp_tiles(const float *a, const float *b, float *c,
                            size_t batch, size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 10 — OpenMP over output tiles
       Same batched/shared-weight layout. Parallelize independent (t,ii,jj)
       output tiles. Each worker owns its C tile and completes the full q/K
       reduction locally; do not split that reduction across workers, which
       would require synchronization and could race on partial sums.
       Pseudocode: parallel for each (t,ii,jj): for each output in that tile,
       compute its full sum over q=0..K-1 and store exactly once. */
    /* The placeholder does not read these inputs/options yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    /* (void)a;
    (void)b;
    (void)k;
    (void)tile; */
   	#pragma omp parallel for collapse(3)
    for (size_t t = 0; t < batch; ++t) {
		for (size_t ii = 0; ii < m/tile + 1; ++ii) {
			for (size_t jj = 0; jj < n/tile + 1; ++jj) {
				for (size_t i = ii * tile; i < min_size((ii + 1) * tile, m); ++i) {
					for (size_t j = jj * tile; j < min_size((jj + 1) * tile, n); ++j) {
						float sum = 0.0f;
						for (size_t q = 0; q < k; ++q) {
							sum += a[(t * m + i) * k + q] * b[q * n + j];
						}
						c[(t * m + i) * n + j] = sum;
					}
				}
			}
		}
    }
}
