#include "common.h"

__device__ static void write_wcode(// output
                                   uint *d_output,
                                   // input
                                   ull wcode_pos,
                                   uint *s_warpcode,
                                   uint wcode_len,
                                   int warp_lane) {
	if (wcode_len > 0) {
		// Calculate a pointer to the sub-vector of d_output in which
		// the warp-code will be written
		uint *d_warpcode = d_output + (wcode_pos / 32);

		// Compute the index of the last element of d_warpcode
		// that will be written
		uint aux_len = (wcode_pos % 32) + wcode_len;
		uint last_idx = (aux_len / 32) - ((aux_len % 32) ? 0 : 1);

		// Copy the content of s_warpcode to d_warpcode coalescently
		for (int out_idx = warp_lane; out_idx <= last_idx; out_idx += WARP_SIZE) {
			if ((out_idx == 0) || (out_idx == last_idx))
				atomicOr(&(d_warpcode[out_idx]), s_warpcode[out_idx]);
			else
				d_warpcode[out_idx] = s_warpcode[out_idx];
		}
	}
}

