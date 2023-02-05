#include "sum_field_macros.h"

__device__ static void get_sum_of_prev_values_within_segment(// output
															 ull &sum_of_prev_values_within_segment,
															 // input
															 uint segment_idx,
															 int segment_lane,
															 volatile ull *d_scan) {
	sum_of_prev_values_within_segment = 0;

	if (segment_lane > 0) {
		volatile ull *d_segment = d_scan + segment_idx * WARP_SIZE;
		uint prev_segment_lane = segment_lane - 1;

		// Read the previous element of the segment repeatedly until all the corresponding sums
		// have been performed on it (1 if it is the first element of the segment,
		// 2 if it is the second, and so on)
		while (1) {
			sum_of_prev_values_within_segment = d_segment[prev_segment_lane];
			uint num_of_sums_in_prev_elem = GET_SUM_FIELD(sum_of_prev_values_within_segment);

			if (ALL_SUMS_PERFORMED(num_of_sums_in_prev_elem, prev_segment_lane)) {
				break;
			}
		}

		sum_of_prev_values_within_segment = DEL_SUM_FIELD(sum_of_prev_values_within_segment);
	}
}


