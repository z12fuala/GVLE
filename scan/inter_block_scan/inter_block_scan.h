#include "calc_segment_param.h"
#include "sum_value_to_segment.h"
#include "get_sum_of_prev_values_within_segment.h"
#include "get_sum_of_prev_segments.h"
#include "write_sum_of_segments_0_to_current.h"

// Exclusive inter-block scan
__device__ static void exc_inter_block_scan(// output
											volatile ull &s_prefix_sum,
											// input
											uint value,
											uint value_idx,
											int warp_lane,
											int warp_idx,
											volatile ull *d_scan,
											uint num_values) {
	int segment_idx, segment_lane, segment_last_lane;
	calc_segment_param(// output
					   segment_idx, segment_lane, segment_last_lane,
					   // input
					   value_idx, num_values);

	ull segment_sum;
	sum_value_to_segment(// output
						 d_scan, segment_sum,
						 // input
						 value, segment_idx, segment_lane,
						 warp_lane, segment_last_lane,
						 num_values);

	ull sum_of_prev_values_within_segment;
	get_sum_of_prev_values_within_segment(// output
										  sum_of_prev_values_within_segment,
										  // input
										  segment_idx, segment_lane,
										  d_scan);

	ull sum_of_prev_segments;
	get_sum_of_prev_segments(// output
							 sum_of_prev_segments,
							 // input
							 d_scan, segment_idx);

	s_prefix_sum = sum_of_prev_values_within_segment + sum_of_prev_segments;

	if (segment_idx > 0 && segment_sum > 0) {
		ull sum_of_segments_0_to_current = sum_of_prev_segments + segment_sum;

		write_sum_of_segments_0_to_current(// output
										   d_scan,
										   // input    
										   segment_idx,
										   sum_of_segments_0_to_current);
	}
}



