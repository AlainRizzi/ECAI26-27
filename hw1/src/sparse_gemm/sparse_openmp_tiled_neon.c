#include "sparse_gemm.h"
#include <stdio.h>
#include <arm_neon.h>

int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;
    /* TODO: combine the row-tile reuse in gemm_sparse_tiled with the
       four-column NEON update in gemm_sparse_neon. For each independent row
       tile, keep one float32x4_t accumulator per row and output-column group.
       Load each retained 4-value weight row once, then update all rows in the
       four-row tile with vfmaq_f32. Parallelize only the outer row-tile loop
       with a plain '#pragma omp parallel for'. The row tile is fixed at four;
       the benchmark tests thread counts 1/2/4. Use the same packed block
       traversal and ordering as the other kernels. */
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < m; i+=4){
        for(size_t g = 0; g < w->n/4; ++g){
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            for(size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g+1]; ++p){
                size_t k0 = 4 * (size_t)w->block_row_idx[p];
                for (size_t r = 0; r < 4; ++r) {
                    float32x4_t A1 = vdupq_n_f32(a[i * w->k + k0 + r]);
                    float32x4_t A2 = vdupq_n_f32(a[(i+1) * w->k + k0 + r]);
                    float32x4_t A3 = vdupq_n_f32(a[(i+2) * w->k + k0 + r]);
                    float32x4_t A4 = vdupq_n_f32(a[(i+3) * w->k + k0 + r]);
                    float32x4_t W = vld1q_f32(&w->block_values[16 * p + 4 * r]);
                    acc1 = vfmaq_f32(acc1, A1, W);
                    acc2 = vfmaq_f32(acc2, A2, W);
                    acc3 = vfmaq_f32(acc3, A3, W);
                    acc4 = vfmaq_f32(acc4, A4, W);
                }
            }
            vst1q_f32(&c[i * w->n + 4 * g], acc1);
            vst1q_f32(&c[(i+1) * w->n + 4 * g], acc2);
            vst1q_f32(&c[(i+2) * w->n + 4 * g], acc3);
            vst1q_f32(&c[(i+3) * w->n + 4 * g], acc4);
        }
    }
    return SPARSE_OK;
}
