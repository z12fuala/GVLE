#include "common.h"

#include "scan/intra_warp_scan.h"
#include "scan/inter_block_scan/inter_block_scan.h"

template <uint TBLOCK_SIZE>
__device__ static void calc_wcode_param(// output
                                        uint &pos_of_thcode_in_wcode,
                                        uint &wcode_len,
                                        uint &wcode_pos,
                                        // input
                                        uint thcode_len,
                                        ull *d_scan,
                                        uint blockinput_idx,
                                        uint num_blockinputs,
                                        int warp_idx, int warp_lane) {
	// Calculate the bit-positions of the thread-codes in their warp-code
	// and the bit-length of the warp-code	
	exc_warp_scan<WARP_SIZE>(// output
                                 pos_of_thcode_in_wcode, wcode_len,
                                 // input
                                 thcode_len, warp_lane);

	// Write the bit-length of the warp-code to shared memory
	__shared__ uint s_scan[NUM_WARPS(TBLOCK_SIZE)];
	s_scan[warp_idx] = wcode_len;

	__syncthreads();

	__shared__ ull s_blockcode_pos;

	if (warp_idx == 0) {
		// Calculate the bit-positions of the warp-codes in their block-code
		// and the bit-length of the block-code
		uint wcode_len = s_scan[warp_lane];
		uint wcode_len_inc_scan;
		uint blockcode_len;
		inc_warp_scan<NUM_WARPS(TBLOCK_SIZE)>(// output
                                                      wcode_len_inc_scan, blockcode_len,
                                                      // input
                                                      wcode_len, warp_lane);

		// Write the bit-positions calculated in previous step to shared memory
		if (warp_lane < NUM_WARPS(TBLOCK_SIZE)) {
			s_scan[warp_lane] = wcode_len_inc_scan;
		}

		// Calculate the bit-position of the blockcode in the output vector
		exc_inter_block_scan(// output
                                     s_blockcode_pos,
                                     // input
                                     blockcode_len, blockinput_idx,
                                     threadIdx.x, warp_idx,
                                     d_scan, num_blockinputs);
	}

	__syncthreads();

	// Read the bit-position of the warp-code in the block-code from shared memory
	uint pos_of_warpcode_in_blockcode;
	if (warp_idx == 0) {
		pos_of_warpcode_in_blockcode = 0;
	}
	else {
		pos_of_warpcode_in_blockcode = s_scan[warp_idx - 1];
	}

	// Calculate the bit-position of the warp-code in the output vector
	wcode_pos = s_blockcode_pos + pos_of_warpcode_in_blockcode;
}

