#include "../../common.h"

__device__ void static calc_segment_param(// output
                                          int &segment_idx,
                                          int &segment_lane,
                                          int &segment_last_lane,
                                          // input
                                          uint value_idx,
                                          uint num_values) {
	segment_idx = value_idx / WARP_SIZE;

	segment_lane = value_idx % WARP_SIZE;

	int last_segment_idx = (num_values - 1) / WARP_SIZE;
	uint num_of_segment_sums;
	if (segment_idx != last_segment_idx) {
		num_of_segment_sums = WARP_SIZE;
	}
	else {
		uint aux = num_values % WARP_SIZE;
		num_of_segment_sums = (aux == 0 ? WARP_SIZE : aux);
	}
	segment_last_lane = num_of_segment_sums - 1;
}
