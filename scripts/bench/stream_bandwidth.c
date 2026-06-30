/* Achievable DRAM bandwidth, via a STREAM-triad over arrays far larger than cache.
 * This is the "what can the RAM actually sustain" number in docs/01 / docs/05.
 *
 *   build (gcc/clang):  cc -O3 -fopenmp stream_bandwidth.c -o stream
 *   build (MSVC):       cl /O2 /openmp /arch:AVX2 stream_bandwidth.c
 *   run a sweep:        for t in 1 4 8 16 24; do OMP_NUM_THREADS=$t ./stream; done
 *
 * 100M doubles/array = 800 MB each, 2.4 GB total, well past any L3, so it's pure DRAM traffic.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    int N = 100 * 1000 * 1000, NTIMES = 12, i, k;
    double scalar = 3.0, best = 1e30;
    double *a = (double*)malloc((size_t)N * 8);
    double *b = (double*)malloc((size_t)N * 8);
    double *c = (double*)malloc((size_t)N * 8);
    if (!a || !b || !c) { printf("alloc fail\n"); return 1; }

    #pragma omp parallel for
    for (i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }

    for (k = 0; k < NTIMES; k++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for
        for (i = 0; i < N; i++) a[i] = b[i] + scalar * c[i];
        double dt = omp_get_wtime() - t0;
        if (dt < best) best = dt;
        if (a[k % N] < 0.0) printf("x");   /* defeat dead-code elimination */
    }
    printf("threads=%2d  triad BW = %.1f GB/s\n", omp_get_max_threads(), 3.0 * N * 8 / best / 1e9);
    free(a); free(b); free(c);
    return 0;
}
