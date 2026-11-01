#include "sparse_gemm.h"
#include <stdio.h>

int gemm_sparse_tiled(const float *a, const sparse_weights *w, float *c,
                        size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;
    /* TODO: use a row tile of exactly four input rows. Keep four 4-column
       output accumulators in local storage. For every retained block, load
       its weight values once and reuse them for all four rows. */
    for (size_t i = 0; i < m; i += 4) {
        for (size_t g = 0; g < w->n / 4; ++g) {
            float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float acc3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                size_t k0 = 4 * (size_t)w->block_row_idx[p];
                for (size_t r = 0; r < 4; ++r) {
                    float av1 = a[(i) * w->k + k0 + r];
                    float av2 = a[(i + 1) * w->k + k0 + r];
                    float av3 = a[(i + 2) * w->k + k0 + r];
                    float av4 = a[(i + 3) * w->k + k0 + r];
                    for (size_t col = 0; col < 4; ++col) {
                        float wv = w->block_values[16 * p + 4 * r + col];
                        acc1[col] += av1 * wv;
                        acc2[col] += av2 * wv;
                        acc3[col] += av3 * wv;
                        acc4[col] += av4 * wv;
                    }
                }
            }
            for (size_t col = 0; col < 4; ++col) {
                c[(i) * w->n + 4 * g + col] = acc1[col];
                c[(i + 1) * w->n + 4 * g + col] = acc2[col];
                c[(i + 2) * w->n + 4 * g + col] = acc3[col];
                c[(i + 3) * w->n + 4 * g + col] = acc4[col];
            }   
        }
    }
    return SPARSE_OK;
}
