#include "railgun.h"

#include <numeric>
#include <iostream>
#include <curand_kernel.h>

__global__
void monte_pi(int seed, int n_try, double* out) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState s;

    curand_init((unsigned int)seed, id, 0, &s);

    int inside = 0;
    for (int i = 0; i < n_try; ++i) {
        double x = curand_uniform(&s);
        double y = curand_uniform(&s);
        if (x * x + y * y < 1) {
            inside++;
        }
    }

    out[id] = inside / static_cast<double>(n_try);
}

int main() {
    int const n_thread = 256;
    int const n_block = 256;
    int const n = n_thread * n_block;
    int const n_try = 1000;

    double *out;
    cudaHostAlloc((void**)&out, n*sizeof(double), cudaHostAllocDefault);

    railgun_t *rg;
    railgun_args *args;

    rg = get_railgun();
    args = rg->wrap_args("II|d", 0, 1, n_try, 1, out, n);
    rg->schedule((void*)monte_pi, args, n_thread, n_block);
    rg->execute();

    std::cout
        << 4.0 * std::accumulate(out, out + n, 0.0) / n
        << std::endl;

    cudaFreeHost(out);

    return 0;
}
