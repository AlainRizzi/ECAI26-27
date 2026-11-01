#include "sparse_gemm.h"
#include <stdio.h>

int gemm_sparse_openmp(const float *a, const sparse_weights *w, float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;
    /* TODO: parallelize only the independent output-row loop. Each iteration
       must own a distinct row of C and compute that row's 4-column groups. */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m; ++i) {
        /* TODO: compute output row i with the scalar sparse loop. */
        gemm_sparse_scalar(&a[i * w->k], w, &c[i * w->n], 1);
    }
    return SPARSE_OK;
}
