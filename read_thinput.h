#include "common.h"

__device__ inline uchar32 read_thinput(uint *d_input, uint thinput_idx) {
	// Read thread-input of index thinput_idx 
	// through one vectorized access
	return ((uchar32 *)d_input)[thinput_idx];
}


