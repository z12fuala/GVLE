#include "sum_field_macros.h"

__device__ static void sum_value_to_segment(// output
											volatile ull *d_scan,
											ull &segment_sum,
											// input											 
											uint value,
											int segment_idx,
											int segment_lane,
											int warp_lane,
											int segment_last_lane,
											uint num_values) {
	// Initialize sum field to 1
	ull flagged_value = INIT_SUM_FIELD(value);

	// Sum flagged value to elements warp_lane, warp_lane + 1, .... of current segment atomically.
	// Note that the sums counter of each element is updated automatically.
	ull prev_value = 0;
	uint target_value_idx = segment_idx * WARP_SIZE + warp_lane;
	if (warp_lane >= segment_lane
		&&
		target_value_idx < num_values
		) {
		prev_value = atomicAdd(&((ull *)d_scan)[target_value_idx], flagged_value);
	}

	// If all sums of the segment have been performed, compute the total sum of the segment
	uint num_of_additions_made = GET_SUM_FIELD(prev_value) + 1;
	if ((warp_lane == segment_last_lane) && (ALL_SUMS_PERFORMED(num_of_additions_made, segment_last_lane))) {
		segment_sum = DEL_SUM_FIELD(prev_value) + (ull)value;
	}
	else
		segment_sum = 0;
}

