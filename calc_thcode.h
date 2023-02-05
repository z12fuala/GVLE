#include "common.h"

__device__ static void calc_thcode(// output
                                   uint codeword_val[16],
                                   uint codeword_len[16],
                                   uint &thcode_len,
                                   // input
                                   uchar32 thinput,
                                   ushort s_VLET_val[256],
                                   uchar s_VLET_len[256],
                                   uint idx_of_first_symbol_of_thinput,
                                   uint num_symbols) {
	uint cw_len[32];
	uint cw_val[32];

	thcode_len = 0;
	#pragma unroll
	for (uint i = 0; i < 32; i += 2) {
		// Read codeword assigned to symbol i of thread-input
		bool valid_symbol_0 = (idx_of_first_symbol_of_thinput + i) < num_symbols;
		cw_len[i] = (valid_symbol_0 ? s_VLET_len[thinput.symbol[i]] : 0);
		cw_val[i] = (valid_symbol_0 ? s_VLET_val[thinput.symbol[i]] : 0);

		// Read codeword assigned to symbol i + 1 of thread-input
		bool valid_symbol_1 = (idx_of_first_symbol_of_thinput + i + 1) < num_symbols;
		cw_len[i + 1] = (valid_symbol_1 ? s_VLET_len[thinput.symbol[i + 1]] : 0);
		cw_val[i + 1] = (valid_symbol_1 ? s_VLET_val[thinput.symbol[i + 1]] : 0);

		// Concatenate codewords
		codeword_val[i / 2] = (cw_val[i] << cw_len[i + 1]) + cw_val[i + 1];
		codeword_len[i / 2] = cw_len[i] + cw_len[i + 1];

		// Update bit-length of thread-code
		thcode_len += codeword_len[i / 2];
	}
}


