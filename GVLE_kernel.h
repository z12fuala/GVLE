#include "cache_VLET.h"
#include "get_global_counter_value.h"
#include "read_thinput.h"
#include "calc_thcode.h"
#include "calc_wcode_param.h"
#include "build_wcode.h"
#include "write_wcode.h"

template <uint TBLOCK_SIZE = 128>
__global__ static void GVLE_kernel(// output
				   uint *d_output,
				   // input
				   uint *d_input,
				   uint num_symbols,
				   ushort d_VLET_val[256],
				   uchar d_VLET_len[256],
				   ull *d_scan,
				   uint d_global_counter[1]) {
	// Initializations
	const uint num_blockinputs = CEIL_DIV(num_symbols, TBLOCK_SIZE * 32);
	const uint last_thinput_idx = CEIL_DIV(num_symbols, 32) - 1;
	const int warp_idx = WARP_IDX(threadIdx.x);
	const int warp_lane = WARP_LANE(threadIdx.x);

	// Cache VLET in shared memory
	__shared__ ushort s_VLET_val[256];
	__shared__ uchar s_VLET_len[256];
	cache_VLET<TBLOCK_SIZE>(s_VLET_val, s_VLET_len, d_VLET_val, d_VLET_len);

	// Get index of first block-input to encode
	uint blockinput_idx = get_global_counter_value(d_global_counter);

	// While there are block-inputs to encode...
	while (blockinput_idx < num_blockinputs) {
		// Initializations		
		uint thinput_idx = blockinput_idx * TBLOCK_SIZE + threadIdx.x;
		uint idx_of_first_symbol_of_thinput = thinput_idx * 32;
		bool is_last_thinput = (thinput_idx == last_thinput_idx);

		// Read thread-input		
		uchar32 thinput = read_thinput(d_input, thinput_idx);

		// Calculate thread-code		
		uint seg_val[16];
		uint seg_len[16];
		uint thcode_len;
		calc_thcode(// output
			    seg_val, seg_len, thcode_len,
			    // input
			    thinput, s_VLET_val, s_VLET_len,
			    idx_of_first_symbol_of_thinput,
			    num_symbols);

		// Calculate parameters of warp-code
		uint pos_of_thcode_in_wcode, wcode_len, wcode_pos;
		calc_wcode_param<TBLOCK_SIZE>(// output
					      pos_of_thcode_in_wcode, 
			                      wcode_len, wcode_pos,
					      // input
					      thcode_len, d_scan,
					      blockinput_idx, num_blockinputs,
					      warp_idx, warp_lane);

		// Build warp-code in shared memory		
		__shared__ uint s_warpcodes[NUM_WARPS(TBLOCK_SIZE)]
			                   [WARP_SIZE * 16 + 1];
		build_wcode(// output
			    s_warpcodes[warp_idx],
			    // input
			    seg_val, seg_len, thcode_len,
			    pos_of_thcode_in_wcode, wcode_pos,
			    idx_of_first_symbol_of_thinput,
			    num_symbols,
			    warp_lane, is_last_thinput);

		// Write warp-code to output vector
		write_wcode(// output
			    d_output,
			    // input
			    wcode_pos, s_warpcodes[warp_idx],
			    wcode_len, warp_lane);

		// Get index of next block-input to encode
		blockinput_idx = get_global_counter_value(d_global_counter);
	}
}


