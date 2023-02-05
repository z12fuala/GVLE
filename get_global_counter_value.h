#include "common.h"

__shared__ static uint s_blockinput_idx;

__device__ inline uint get_global_counter_value(uint *d_global_counter) {
	if (threadIdx.x == 0) {
		s_blockinput_idx = atomicAdd(d_global_counter, 1);
	}
	__syncthreads();

	return s_blockinput_idx;
}

