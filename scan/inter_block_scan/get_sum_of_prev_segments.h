#include "sum_field_macros.h"

__device__ static void get_sum_of_prev_segments(// output
                                                ull &prev_segments_sum,
                                                // input
                                                volatile ull *d_scan,
                                                int segment_idx) {
	prev_segments_sum = 0;

	if (segment_idx > 0) {
		// For each previous segment, from the last to the first...
		for (int prev_segment_idx = (segment_idx - 1); prev_segment_idx >= 0; prev_segment_idx--) {
			// Index of the last element of current previous segment
			int prev_value_idx = prev_segment_idx * WARP_SIZE + WARP_SIZE - 1;

			// Read the last element of current previous segment repeatedly
			// until one of the following conditions is met:
			// - The flag P is set, which means that the element stores the
			//   sum of the current previous segment and all the remaining
			//   previous segments.
			// - The element holds the sum of the segment.
			ull prev_segment_value;
			while (1) {
				prev_segment_value = d_scan[prev_value_idx];
				uint sum_field = GET_SUM_FIELD(prev_segment_value);

				if (sum_field >= WARP_SIZE) {
					break;
				}
			}

			// Update the sum of previous segments
			prev_segments_sum += DEL_SUM_FIELD(prev_segment_value);

			// If the last read sum correspond to the current previous segment
			// and all the remaining previous segments, finish
			if (FLAG_P_IS_SET(prev_segment_value)) {
				break;
			}
		}
	}
}
