#include "common.h"

template <uint TBLOCK_SIZE>
__device__ static void cache_VLET(// output
								  ushort s_VLET_val[256],
								  uchar s_VLET_len[256],
								  // input  
								  ushort d_VLET_val[256],
								  uchar d_VLET_len[256]) {
	// Copy d_VLET_val/d_VLET_len to s_VLET_val/s_VLET_len
	// coalescently
	for (int i = threadIdx.x; i < 256; i += TBLOCK_SIZE) {
		s_VLET_val[i] = d_VLET_val[i];
		s_VLET_len[i] = d_VLET_len[i];
	}

	__syncthreads();
}


