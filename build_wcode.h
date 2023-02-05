#include "common.h"

__device__ static void build_wcode(// output
								   uint *s_warpcode,
								   // input	
								   uint seg_val[16],
								   uint seg_len[16],
								   uint thcode_len,
								   uint pos_of_thcode_in_warpcode,
								   ull wcode_pos,
								   uint idx_of_first_symbol_of_thinput,
								   uint num_symbols,
								   int warp_lane,
								   bool is_last_thinput) {
	// Compute s_thcode, which is a pointer to the sub-vector of s_warpcode
	// in which the thread-code will be written
	uint buff_wcode_pos = wcode_pos % 32;
	uint buff_thcode_pos = buff_wcode_pos + pos_of_thcode_in_warpcode;
	uint *s_thcode = s_warpcode + (buff_thcode_pos / 32);

	// Write the first chunk and all the 32-bits elements of s_thcode
	uint elem_idx = 0;
	uint elem_value = 0;
	uint elem_len = buff_thcode_pos % 32;
	if (thcode_len > 0) {
		#pragma unroll
		for (uint i = 0; i < 16; i++) {
			uint aux = elem_len + seg_len[i];

			if (aux < 32) {
				elem_value |= seg_val[i] << (32 - aux);
				elem_len = aux;
			}
			else {
				uint new_elem_value = elem_value | (seg_val[i] >> (aux - 32));

				s_thcode[elem_idx] = new_elem_value;

				elem_idx++;
				elem_value = seg_val[i] << (64 - aux);
				elem_len = aux - 32;
			}
		}

		// Special case: the bit-length of the last thread-code
		// of the output vector is fewer than 32 bits
		if (elem_idx == 0) {
			s_thcode[elem_idx] = elem_value;
			elem_len = 0;
		}
	}

	__syncwarp();

	// Write the last chunk of s_thcode, if it exists
	if (thcode_len > 0 && elem_len > 0) {
		bool store_elem =
			(warp_lane == (WARP_SIZE - 1)) || is_last_thinput;

		if (store_elem) {
			s_thcode[elem_idx] = elem_value;
		}
		else {
			s_thcode[elem_idx] |= elem_value;
		}
	}

	__syncwarp();
}






